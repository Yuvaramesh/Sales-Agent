import sys
import time
from typing import Dict, Any, List, Optional
from memory_optimizer import MemoryOptimizerMixin
from langgraph_memory import (
    LANGGRAPH_AVAILABLE,
    InMemorySaver,
    InMemoryStore,
    HumanMessage,
    AIMessage,
)
from database import convos_col, users_col, summaries_col, failed_writes_col
from utils import utcnow_iso, sanitize_text, robust_extract_content
from llm_setup import llm


class ConversationMemoryManager(MemoryOptimizerMixin):
    """Manages conversation state, history, and persistence"""

    def __init__(self):
        super().__init__()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        if LANGGRAPH_AVAILABLE:
            try:
                self.checkpointer = InMemorySaver()
                self.store = InMemoryStore()
            except Exception:
                self.checkpointer = None
                self.store = None
        else:
            self.checkpointer = None
            self.store = None

    def _new_session(self, user_email: str) -> Dict[str, Any]:
        """Create a new session object"""
        return {
            "user_email": user_email,
            "start_time": utcnow_iso(),
            "messages": [],
            "stage": "init",
            "collected": {},
            "last_results": [],
            "last_web_results": [],
            "selected_vehicle": None,
            "order_id": None,
            "memory_summary": "",
            "awaiting": None,
        }

    def hydrate_langgraph_memory(self, session_id: str):
        """Load conversation history into LangGraph checkpointer"""
        if not self.checkpointer:
            return
        try:
            rows = list(
                convos_col.find({"session_id": session_id})
                .sort("timestamp", 1)
                .limit(50)
            )
            msgs = []
            for r in rows:
                u = r.get("user_message")
                b = r.get("bot_response")
                if u:
                    msgs.append(HumanMessage(content=u))
                if b:
                    msgs.append(AIMessage(content=b))
            if msgs:
                try:
                    self.checkpointer.put(
                        {"configurable": {"thread_id": session_id}}, {"messages": msgs}
                    )
                except TypeError:
                    self.checkpointer.put(
                        {"configurable": {"thread_id": session_id}},
                        {"messages": msgs},
                        {},
                    )
        except Exception:
            pass

    def ensure_session_loaded(self, session_id: str, user_email: str = "") -> bool:
        """Load session from database if not in memory"""
        if not session_id:
            return False
        if session_id in self.sessions:
            return True

        try:
            # Try to find user with this session
            u = users_col.find_one({"current_session.session_id": session_id})
            if u:
                s = self._new_session(u.get("email") or user_email)
                cs = u.get("current_session", {})
                s["stage"] = cs.get("stage", s["stage"])
                s["selected_vehicle"] = cs.get(
                    "selected_vehicle", s["selected_vehicle"]
                )
                s["order_id"] = cs.get("order_id", s["order_id"])
                s["collected"] = cs.get("collected", s["collected"])
                s["memory_summary"] = cs.get(
                    "memory_summary", s.get("memory_summary", "")
                )
                s["awaiting"] = cs.get("awaiting", s.get("awaiting"))

                # Load conversation history
                try:
                    rows = list(
                        convos_col.find({"session_id": session_id})
                        .sort("timestamp", 1)
                        .limit(200)
                    )
                    if rows:
                        msgs = []
                        for r in rows:
                            msgs.append(
                                {
                                    "user": r.get("user_message"),
                                    "assistant": r.get("bot_response"),
                                    "agent": r.get("agent_used"),
                                    "timestamp": r.get("timestamp"),
                                }
                            )
                        s["messages"] = msgs
                except Exception:
                    pass

                self.sessions[session_id] = s
                try:
                    from helpers import persist_session_state_raw

                    persist_session_state_raw(
                        s.get("user_email", user_email) or user_email, session_id, s
                    )
                except Exception:
                    pass
                return True

            # Try to find by email
            if user_email:
                u2 = users_col.find_one({"email": user_email})
                if (
                    u2
                    and u2.get("current_session")
                    and u2["current_session"].get("session_id")
                ):
                    real_sid = u2["current_session"].get("session_id")
                    if real_sid and real_sid not in self.sessions:
                        return self.ensure_session_loaded(real_sid, user_email)

            # Try to find conversation history
            rows = list(
                convos_col.find({"session_id": session_id})
                .sort("timestamp", 1)
                .limit(200)
            )
            if rows:
                first = rows[0]
                s = self._new_session(first.get("user_email", "") or user_email)
                msgs = []
                for r in rows:
                    msgs.append(
                        {
                            "user": r.get("user_message"),
                            "assistant": r.get("bot_response"),
                            "agent": r.get("agent_used"),
                            "timestamp": r.get("timestamp"),
                        }
                    )
                s["messages"] = msgs
                self.sessions[session_id] = s
                try:
                    from helpers import persist_session_state_raw

                    persist_session_state_raw(
                        s.get("user_email", user_email) or user_email, session_id, s
                    )
                except Exception:
                    pass
                return True

            # Last resort: find any conversation by email
            if user_email:
                r = convos_col.find_one(
                    {"user_email": user_email}, sort=[("timestamp", -1)]
                )
                if r and r.get("session_id"):
                    return self.ensure_session_loaded(r.get("session_id"), user_email)
        except Exception as e:
            print("[ensure_session_loaded error]", e, file=sys.stderr)
        return False

    def get_or_create_session(
        self, user_email: str, session_id: Optional[str] = None
    ) -> str:
        """Get existing session or create new one"""
        if session_id:
            try:
                loaded = self.ensure_session_loaded(session_id, user_email)
                if not loaded:
                    self.sessions[session_id] = self._new_session(user_email)
                    from helpers import persist_session_state_raw

                    persist_session_state_raw(
                        user_email, session_id, self.sessions[session_id]
                    )

                # Sync with database
                try:
                    u = users_col.find_one({"email": user_email})
                    if (
                        u
                        and "current_session" in u
                        and u["current_session"].get("session_id") == session_id
                    ):
                        cs = u["current_session"]
                        s = self.sessions[session_id]
                        s["stage"] = cs.get("stage", s["stage"])
                        s["selected_vehicle"] = cs.get(
                            "selected_vehicle", s["selected_vehicle"]
                        )
                        s["order_id"] = cs.get("order_id", s.get("order_id"))
                        s["collected"] = cs.get("collected", s.get("collected", {}))
                except Exception:
                    pass
            except Exception:
                self.sessions[session_id] = self._new_session(user_email)
                from helpers import persist_session_state_raw

                persist_session_state_raw(
                    user_email, session_id, self.sessions[session_id]
                )
            return session_id

        # Create new session
        sid = f"{user_email}_{int(time.time())}"
        self.sessions[sid] = self._new_session(user_email)
        from helpers import persist_session_state_raw

        persist_session_state_raw(user_email, sid, self.sessions[sid])
        return sid

    def add_message(
        self, session_id: str, user_message: str, bot_response: str, agent_used: str
    ):
        """Add a message exchange to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._new_session("")

        user_message = sanitize_text(user_message, max_len=4000)
        bot_response = sanitize_text(bot_response, max_len=4000)

        entry = {
            "user": user_message,
            "assistant": bot_response,
            "agent": agent_used,
            "timestamp": utcnow_iso(),
        }
        self.sessions[session_id]["messages"].append(entry)

        # Persist to database
        try:
            conv_doc = {
                "session_id": session_id,
                "user_email": self.sessions[session_id].get("user_email", ""),
                "user_message": user_message,
                "bot_response": bot_response,
                "agent_used": agent_used,
                "timestamp": utcnow_iso(),
                "turn_index": len(self.sessions[session_id]["messages"]) - 1,
            }
            convos_col.insert_one(conv_doc)
        except Exception as e:
            print("[convos_col insert error]", e, file=sys.stderr)
            try:
                failed_writes_col.insert_one(
                    {
                        "collection": "conversations",
                        "error": str(e),
                        "doc": conv_doc,
                        "timestamp": utcnow_iso(),
                    }
                )
            except Exception:
                pass

        # Update LangGraph checkpointer
        if self.checkpointer:
            try:
                config = {"configurable": {"thread_id": session_id}}
                state = {
                    "messages": [
                        HumanMessage(content=user_message),
                        AIMessage(content=bot_response),
                    ]
                }
                try:
                    self.checkpointer.put(config, state, {})
                except TypeError:
                    self.checkpointer.put(config, state)
            except Exception:
                pass

        # Update LangGraph store
        if self.store:
            try:
                namespace = ("conversations", session_id)
                key = f"msg_{len(self.sessions[session_id]['messages'])}"
                try:
                    self.store.put(namespace, key, entry)
                except TypeError:
                    self.store.put(namespace, key, entry, {})
            except Exception:
                pass

        # Persist session state
        try:
            from helpers import persist_session_state

            persist_session_state(session_id)
        except Exception as e:
            print("[persist_session_state error]", e, file=sys.stderr)

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a session"""
        if session_id in self.sessions and self.sessions[session_id]["messages"]:
            return self.sessions[session_id]["messages"]

        # Try database
        try:
            rows = list(
                convos_col.find({"session_id": session_id}).sort("timestamp", 1)
            )
            if rows:
                out = []
                for r in rows:
                    out.append(
                        {
                            "user": r.get("user_message"),
                            "assistant": r.get("bot_response"),
                            "agent": r.get("agent_used"),
                            "timestamp": r.get("timestamp"),
                        }
                    )
                if session_id not in self.sessions:
                    self.sessions[session_id] = self._new_session(
                        rows[0].get("user_email", "")
                    )
                self.sessions[session_id]["messages"] = out
                return out
        except Exception as e:
            print("[get_session_messages error]", e, file=sys.stderr)

        # Try LangGraph store
        if self.store is not None:
            try:
                namespace = ("conversations", session_id)
                items = self.store.search(namespace)
                if items:
                    return [it.value for it in items]
            except Exception:
                pass

        return []

    def generate_summary(self, session_id: str) -> str:
        """Generate a summary of the conversation"""
        msgs = self.get_session_messages(session_id)
        if not msgs:
            return "No messages to summarize."

        convo_text = []
        for m in msgs:
            convo_text.append(f"User: {m.get('user')}")
            convo_text.append(f"Assistant: {m.get('assistant')}")

        prompt = (
            "Summarize the following conversation concisely. Include main topics, "
            "the selected vehicle (if chosen), and next steps.\n\n"
            "Conversation:\n" + "\n".join(convo_text) + "\n\nSummary:"
        )

        if llm:
            try:
                resp = llm([{"role": "user", "content": prompt}])
                summary = robust_extract_content(resp)
            except Exception:
                summary = "Summary (fallback): " + " | ".join(
                    [m.get("user", "")[:80] for m in msgs[:3]]
                )
        else:
            summary = "Summary (fallback): " + " | ".join(
                [m.get("user", "")[:80] for m in msgs[:3]]
            )

        return sanitize_text(summary, max_len=1000)

    def end_session_and_save(self, session_id: str):
        """End session and save summary"""
        if session_id not in self.sessions:
            return "No session messages to summarize."

        summary = self.generate_summary(session_id)
        msgs = self.sessions[session_id]["messages"]
        message_count = len(msgs)
        start_time = self.sessions[session_id].get("start_time")
        end_time = utcnow_iso()
        user_email = self.sessions[session_id].get("user_email", "")

        try:
            summaries_col.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "session_id": session_id,
                        "user_email": user_email,
                        "summary": summary,
                        "message_count": message_count,
                        "start_time": start_time,
                        "end_time": end_time,
                        "created_at": utcnow_iso(),
                    }
                },
                upsert=True,
            )
            if user_email:
                users_col.update_one(
                    {"email": user_email},
                    {
                        "$set": {
                            "recent_summary": summary,
                            "last_session_id": session_id,
                        }
                    },
                    upsert=True,
                )
        except Exception as e:
            print("[end_session_and_save error]", e, file=sys.stderr)

        self.sessions[session_id]["stage"] = "finished"
        try:
            from helpers import persist_session_state

            persist_session_state(session_id)
        except Exception:
            pass

        return summary


# Global instance
memory_manager = ConversationMemoryManager()
