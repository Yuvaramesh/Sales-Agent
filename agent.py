# """
# Final agent.py - Fixed version with proper order placement
# """
# import os
# import re
# import json
# import time
# import sys
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime, timezone, date
# from dotenv import load_dotenv

# load_dotenv()

# MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# if not MONGO_URI:
#     raise RuntimeError("MONGODB_CONNECTION_STRING not set in .env")

# # ---------- MongoDB ----------
# from pymongo import MongoClient
# mongo = MongoClient(MONGO_URI)
# db = mongo.get_database()
# cars_col = db.get_collection("cars")
# users_col = db.get_collection("users")
# convos_col = db.get_collection("conversations")
# summaries_col = db.get_collection("conversation_summaries")
# orders_col = db.get_collection("orders")
# failed_writes_col = db.get_collection("failed_writes")

# # ---------- LangGraph Memory ----------
# try:
#     from langgraph.checkpoint.memory import InMemorySaver
#     from langgraph.store.memory import InMemoryStore
#     from langchain_core.messages import HumanMessage, AIMessage
#     LANGGRAPH_AVAILABLE = True
# except Exception:
#     LANGGRAPH_AVAILABLE = False
#     InMemorySaver = InMemoryStore = None
#     class HumanMessage:
#         def __init__(self, content: str):
#             self.content = content
#     class AIMessage:
#         def __init__(self, content: str):
#             self.content = content

# # ---------- LangChain / OpenAI detection ----------
# llm = None
# LC_AVAILABLE = False
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain.agents import create_agent
#     from langchain.tools import tool
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
#     LC_AVAILABLE = True
# except Exception:
#     import openai
#     openai.api_key = OPENAI_API_KEY
#     class SimpleOpenAIWrapper:
#         def __init__(self, model=LLM_MODEL_NAME, temperature=0):
#             self.model = model
#             self.temperature = temperature
#         def __call__(self, messages: List[Dict[str,str]]):
#             return openai.ChatCompletion.create(model=self.model, messages=messages, temperature=self.temperature)
#     llm = SimpleOpenAIWrapper()
#     LC_AVAILABLE = False

# # ---------- Utilities ----------
# def utcnow_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def sanitize_text(s: str, max_len: int = 4000) -> str:
#     if s is None:
#         return ""
#     if not isinstance(s, str):
#         s = str(s)
#     s = re.sub(r"\s+", " ", s).strip()
#     if len(s) > max_len:
#         return s[:max_len] + "..."
#     return s

# def _make_json_safe(obj: Any) -> Any:
#     try:
#         return json.loads(json.dumps(obj, default=str))
#     except Exception:
#         return str(obj)

# def normalize_vehicle(vehicle):
#     """Ensure vehicle is a dict."""
#     if not vehicle:
#         return None
#     if isinstance(vehicle, dict):
#         return vehicle
#     return None

# def estimate_tokens(text: str) -> int:
#     if not text:
#         return 0
#     return max(1, int(len(text) / 4))

# def extract_contact_info(text: str) -> Dict[str, str]:
#     """Extract contact information from user input"""
#     info = {}

#     # Extract name
#     name_match = re.search(r'name\s*:?\s*([^,\n]+)', text, re.IGNORECASE)
#     if name_match:
#         info['name'] = name_match.group(1).strip()

#     # Extract phone
#     phone_match = re.search(r'phone\s*:?\s*([\+\d\s\-\(\)]+)', text, re.IGNORECASE)
#     if phone_match:
#         info['phone'] = phone_match.group(1).strip()

#     # Extract email
#     email_match = re.search(r'email\s*:?\s*([^\s,]+@[^\s,]+)', text, re.IGNORECASE)
#     if email_match:
#         info['email'] = email_match.group(1).strip()

#     # Extract address (look for address: or delivery address:)
#     address_match = re.search(r'(?:delivery\s+)?address\s*:?\s*([^,]+(?:,[^,]+)*)', text, re.IGNORECASE)
#     if address_match:
#         info['address'] = address_match.group(1).strip()

#     return info

# # ---------- Memory optimizer mixin ----------
# class MemoryOptimizerMixin:
#     MAX_PROMPT_TOKENS = 3000
#     RECENT_TURNS_KEEP = 8
#     SUMMARIZE_EVERY = 12
#     SUMMARY_MAX_TOKENS = 800

#     def compress_history_if_needed(self, session_id: str):
#         s = self.sessions.get(session_id)
#         if not s:
#             return
#         msgs = s.get("messages", [])
#         if len(msgs) <= (self.RECENT_TURNS_KEEP + 2):
#             return
#         last_summary_at = s.get("_last_summary_index", 0)
#         if len(msgs) - last_summary_at < self.SUMMARIZE_EVERY:
#             return
#         older = msgs[: max(0, len(msgs) - self.RECENT_TURNS_KEEP)]
#         if not older:
#             return
#         older_text = []
#         for m in older:
#             u = m.get("user") or ""
#             a = m.get("assistant") or ""
#             if u:
#                 older_text.append(f"User: {sanitize_text(u, 2000)}")
#             if a:
#                 older_text.append(f"Assistant: {sanitize_text(a, 2000)}")
#         to_summarize = "\n".join(older_text)
#         if estimate_tokens(to_summarize) < (self.SUMMARY_MAX_TOKENS // 2):
#             summary = to_summarize
#         else:
#             prompt = ("Summarize the following conversation history into concise bullet points. "
#                       "Keep facts, decisions, selected vehicle details, outstanding questions and next steps. "
#                       "Limit to ~200-400 words.\n\nHistory:\n" + to_summarize + "\n\nSummary:")
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#                 if not summary:
#                     summary = "(summary generation failed)"
#             except Exception as e:
#                 summary = f"(summary generation failed: {e})"
#         prev = s.get('memory_summary', '') or ''
#         new_summary = (prev + '\n---\n' + summary) if prev else summary
#         recent = msgs[-self.RECENT_TURNS_KEEP:]
#         placeholder = {"user": "[older history summarized]", "assistant": new_summary, "agent": "system_summary", "timestamp": utcnow_iso()}
#         s['messages'] = [placeholder] + recent
#         s['_last_summary_index'] = len(s['messages'])
#         s['memory_summary'] = new_summary
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[compress_history persist error]", e, file=sys.stderr)

#     def get_context_for_llm(self, session_id: str, max_messages: int = None) -> str:
#         s = self.sessions.get(session_id)
#         if not s:
#             return ""
#         try:
#             self.compress_history_if_needed(session_id)
#         except Exception:
#             pass
#         memory_summary = s.get('memory_summary', '') or ''
#         recent = s.get('messages', [])[-self.RECENT_TURNS_KEEP:]
#         lines = []
#         tokens_used = 0
#         if memory_summary:
#             ts = f"Memory Summary:\n{memory_summary}\n"
#             t_count = estimate_tokens(ts)
#             lines.append(ts)
#             tokens_used += t_count
#         for m in recent:
#             u = m.get('user') or ''
#             a = m.get('assistant') or ''
#             agent = m.get('agent') or ''
#             if u:
#                 line = f"User: {u}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#             if a:
#                 line = f"Assistant ({agent}): {a}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#         return "\n".join(lines)

# # ---------- Conversation memory manager ----------
# class ConversationMemoryManager(MemoryOptimizerMixin):
#     def __init__(self):
#         super().__init__()
#         self.sessions: Dict[str, Dict[str, Any]] = {}
#         if LANGGRAPH_AVAILABLE:
#             try:
#                 self.checkpointer = InMemorySaver()
#                 self.store = InMemoryStore()
#             except Exception:
#                 self.checkpointer = None
#                 self.store = None
#         else:
#             self.checkpointer = None
#             self.store = None

#     def _new_session(self, user_email: str) -> Dict[str,Any]:
#         return {
#             "user_email": user_email,
#             "start_time": utcnow_iso(),
#             "messages": [],
#             "stage": "init",
#             "collected": {},
#             "last_results": [],
#             "last_web_results": [],
#             "selected_vehicle": None,
#             "order_id": None,
#             "memory_summary": "",
#             "awaiting": None
#         }

#     def hydrate_langgraph_memory(self, session_id: str):
#         if not self.checkpointer:
#             return
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1).limit(50))
#             msgs = []
#             for r in rows:
#                 u = r.get("user_message")
#                 b = r.get("bot_response")
#                 if u:
#                     msgs.append(HumanMessage(content=u))
#                 if b:
#                     msgs.append(AIMessage(content=b))
#             if msgs:
#                 try:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs})
#                 except TypeError:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs}, {})
#         except Exception:
#             pass

#     def get_or_create_session(self, user_email: str, session_id: Optional[str] = None) -> str:
#         if session_id:
#             if session_id not in self.sessions:
#                 self.sessions[session_id] = self._new_session(user_email)
#                 try:
#                     u = users_col.find_one({"email": user_email})
#                     if u and "current_session" in u and u["current_session"].get("session_id") == session_id:
#                         cs = u["current_session"]
#                         s = self.sessions[session_id]
#                         s["stage"] = cs.get("stage", s["stage"])
#                         s["selected_vehicle"] = cs.get("selected_vehicle", s["selected_vehicle"])
#                         s["order_id"] = cs.get("order_id", s["order_id"])
#                         s["collected"] = cs.get("collected", s["collected"])
#                 except Exception:
#                     pass
#                 try:
#                     self.hydrate_langgraph_memory(session_id)
#                 except Exception:
#                     pass
#             return session_id
#         sid = f"{user_email}_{int(time.time())}"
#         self.sessions[sid] = self._new_session(user_email)
#         persist_session_state_raw(user_email, sid, self.sessions[sid])
#         return sid

#     def add_message(self, session_id: str, user_message: str, bot_response: str, agent_used: str):
#         if session_id not in self.sessions:
#             self.sessions[session_id] = self._new_session("")
#         user_message = sanitize_text(user_message, max_len=4000)
#         bot_response = sanitize_text(bot_response, max_len=4000)
#         entry = {"user": user_message, "assistant": bot_response, "agent": agent_used, "timestamp": utcnow_iso()}
#         self.sessions[session_id]["messages"].append(entry)
#         try:
#             conv_doc = {
#                 "session_id": session_id,
#                 "user_email": self.sessions[session_id].get("user_email", ""),
#                 "user_message": user_message,
#                 "bot_response": bot_response,
#                 "agent_used": agent_used,
#                 "timestamp": utcnow_iso(),
#                 "turn_index": len(self.sessions[session_id]["messages"]) - 1
#             }
#             conv_doc_safe = _make_json_safe(conv_doc)
#             convos_col.insert_one(conv_doc_safe)
#         except Exception as e:
#             print("[convos_col insert error]", e, file=sys.stderr)
#             try:
#                 failed_writes_col.insert_one({"collection": "conversations", "error": str(e), "doc": _make_json_safe(conv_doc), "timestamp": utcnow_iso()})
#             except Exception:
#                 pass
#         if self.checkpointer:
#             try:
#                 config = {"configurable": {"thread_id": session_id}}
#                 state = {"messages": [HumanMessage(content=user_message), AIMessage(content=bot_response)]}
#                 try:
#                     self.checkpointer.put(config, state, {})
#                 except TypeError:
#                     self.checkpointer.put(config, state)
#             except Exception:
#                 pass
#         if self.store:
#             try:
#                 namespace = ("conversations", session_id)
#                 key = f"msg_{len(self.sessions[session_id]['messages'])}"
#                 try:
#                     self.store.put(namespace, key, entry)
#                 except TypeError:
#                     self.store.put(namespace, key, entry, {})
#             except Exception:
#                 pass
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[persist_session_state error]", e, file=sys.stderr)

#     def get_session_messages(self, session_id: str) -> List[Dict[str,Any]]:
#         if session_id in self.sessions and self.sessions[session_id]["messages"]:
#             return self.sessions[session_id]["messages"]
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1))
#             if rows:
#                 out = []
#                 for r in rows:
#                     out.append({
#                         "user": r.get("user_message"),
#                         "assistant": r.get("bot_response"),
#                         "agent": r.get("agent_used"),
#                         "timestamp": r.get("timestamp")
#                     })
#                 if session_id not in self.sessions:
#                     self.sessions[session_id] = self._new_session(rows[0].get("user_email",""))
#                 self.sessions[session_id]["messages"] = out
#                 return out
#         except Exception as e:
#             print("[get_session_messages error]", e, file=sys.stderr)
#         if self.store is not None:
#             try:
#                 namespace = ("conversations", session_id)
#                 items = self.store.search(namespace)
#                 if items:
#                     return [it.value for it in items]
#             except Exception:
#                 pass
#         return []

#     def generate_summary(self, session_id: str) -> str:
#         msgs = self.get_session_messages(session_id)
#         if not msgs:
#             return "No messages to summarize."
#         convo_text = []
#         for m in msgs:
#             convo_text.append(f"User: {m.get('user')}")
#             convo_text.append(f"Assistant: {m.get('assistant')}")
#         prompt = ("Summarize the following conversation concisely. Include main topics, the selected vehicle (if chosen), and next steps.\n\n"
#                   "Conversation:\n" + "\n".join(convo_text) + "\n\nSummary:")
#         if llm:
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#             except Exception:
#                 summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         else:
#             summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         return sanitize_text(summary, max_len=1000)

#     def end_session_and_save(self, session_id: str):
#         if session_id not in self.sessions:
#             return "No session messages to summarize."
#         summary = self.generate_summary(session_id)
#         msgs = self.sessions[session_id]["messages"]
#         message_count = len(msgs)
#         start_time = self.sessions[session_id].get("start_time")
#         end_time = utcnow_iso()
#         user_email = self.sessions[session_id].get("user_email","")
#         try:
#             summaries_col.update_one(
#                 {"session_id": session_id},
#                 {"$set": {
#                     "session_id": session_id,
#                     "user_email": user_email,
#                     "summary": summary,
#                     "message_count": message_count,
#                     "start_time": start_time,
#                     "end_time": end_time,
#                     "created_at": utcnow_iso()
#                 }}, upsert=True
#             )
#             if user_email:
#                 users_col.update_one({"email": user_email},
#                                      {"$set": {"recent_summary": summary, "last_session_id": session_id}},
#                                      upsert=True)
#         except Exception as e:
#             print("[end_session_and_save error]", e, file=sys.stderr)
#         self.sessions[session_id]["stage"] = "finished"
#         try:
#             persist_session_state(session_id)
#         except Exception:
#             pass
#         return summary

# # ---------- Order helpers ----------
# def create_order_with_address(
#     session_id: str,
#     buyer_name: Optional[str] = None,
#     vehicle: Optional[Dict[str, Any]] = None,
#     sales_contact: Optional[Dict[str, Any]] = None,
#     buyer_address: Optional[str] = None,
#     buyer_phone: Optional[str] = None,
#     buyer_email: Optional[str] = None,
# ) -> Optional[str]:
#     session = memory_manager.sessions.get(session_id)
#     if not session:
#         raise ValueError("Invalid session_id")

#     # Get buyer details from collected info
#     collected = session.get("collected", {})
#     if not buyer_name:
#         buyer_name = collected.get("name") or session.get("user_email")
#     if not buyer_address:
#         buyer_address = collected.get("address")
#     if not buyer_phone:
#         buyer_phone = collected.get("phone")
#     if not buyer_email:
#         buyer_email = collected.get("email") or session.get("user_email")

#     vehicle = normalize_vehicle(vehicle) or normalize_vehicle(session.get("selected_vehicle"))

#     if not vehicle:
#         raise ValueError("No vehicle selected for order")
#     if not buyer_address:
#         raise ValueError("Buyer address is required to place order")
#     if not isinstance(vehicle, dict):
#         raise ValueError("Selected vehicle data is invalid.")

#     order_doc = {
#         "session_id": session_id,
#         "user_email": session.get("user_email"),
#         "buyer_name": buyer_name,
#         "buyer_address": buyer_address,
#         "buyer_phone": buyer_phone,
#         "buyer_email": buyer_email,
#         "vehicle": {
#             "make": vehicle.get("make"),
#             "model": vehicle.get("model"),
#             "year": vehicle.get("year"),
#             "price": vehicle.get("price"),
#             "mileage": vehicle.get("mileage"),
#         },
#         "sales_contact": sales_contact or {
#             "name": "Jeni Flemin",
#             "position": "CEO",
#             "phone": "+94778540035",
#             "address": "Convent Garden, London, UK",
#         },
#         "timestamp": utcnow_iso(),
#         "order_date": date.today().isoformat(),
#         "conversation_summary": session.get("memory_summary", ""),
#     }

#     print(f"[create_order] Attempting to insert order: {json.dumps(order_doc, default=str, indent=2)}", file=sys.stderr)

#     try:
#         result = orders_col.insert_one(_make_json_safe(order_doc))
#         order_id = str(result.inserted_id)
#         session["order_id"] = order_id
#         session["stage"] = "ordered"
#         persist_session_state(session_id)
#         print(f"[create_order] Order created successfully with ID: {order_id}", file=sys.stderr)
#         return order_id
#     except Exception as e:
#         print(f"[create_order] Failed to insert order: {e}", file=sys.stderr)
#         try:
#             failed_writes_col.insert_one({
#                 "collection": "orders",
#                 "error": str(e),
#                 "doc": _make_json_safe(order_doc),
#                 "timestamp": utcnow_iso()
#             })
#         except Exception as e2:
#             print(f"[create_order] Failed to log to failed_writes: {e2}", file=sys.stderr)
#         raise

# memory_manager = ConversationMemoryManager()

# # ---------- Helpers ----------
# def persist_session_state(session_id: str):
#     s = memory_manager.sessions.get(session_id)
#     if not s:
#         return
#     email = s.get("user_email", "")
#     try:
#         users_col.update_one(
#             {"email": email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": s.get("stage"),
#                     "selected_vehicle": _make_json_safe(s.get("selected_vehicle")),
#                     "order_id": s.get("order_id"),
#                     "memory_summary": s.get("memory_summary", ""),
#                     "collected": _make_json_safe(s.get("collected", {})),
#                     "awaiting": s.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state]", e, file=sys.stderr)

# def persist_session_state_raw(user_email: str, session_id: str, session_obj: Dict[str,Any]):
#     try:
#         users_col.update_one(
#             {"email": user_email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": session_obj.get("stage"),
#                     "selected_vehicle": _make_json_safe(session_obj.get("selected_vehicle")),
#                     "order_id": session_obj.get("order_id"),
#                     "memory_summary": session_obj.get("memory_summary", ""),
#                     "collected": _make_json_safe(session_obj.get("collected", {})),
#                     "awaiting": session_obj.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": user_email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state_raw]", e, file=sys.stderr)

# def fetch_user_profile_by_email(email: str) -> str:
#     if not email:
#         return "No email provided."
#     p = users_col.find_one({"email": email})
#     if not p:
#         return f"No profile found for {email}."
#     return f"Name: {p.get('name','')}\nEmail: {p.get('email','')}\nRecent summary: {p.get('recent_summary')}"

# def fetch_cars_by_filters(filters: Dict[str,Any], limit: int = 10) -> List[Dict[str,Any]]:
#     q = {}
#     if "make" in filters:
#         q["make"] = {"$regex": re.compile(filters["make"], re.I)}
#     if "model" in filters:
#         q["model"] = {"$regex": re.compile(filters["model"], re.I)}
#     if "year_min" in filters or "year_max" in filters:
#         yq = {}
#         if "year_min" in filters: yq["$gte"] = int(filters["year_min"])
#         if "year_max" in filters: yq["$lte"] = int(filters["year_max"])
#         q["year"] = yq
#     if "price_min" in filters or "price_max" in filters:
#         pq = {}
#         if "price_min" in filters: pq["$gte"] = float(filters["price_min"])
#         if "price_max" in filters: pq["$lte"] = float(filters["price_max"])
#         q["price"] = pq
#     if "mileage_max" in filters:
#         q["mileage"] = {"$lte": int(filters["mileage_max"]) }
#     if "style" in filters:
#         q["style"] = {"$regex": re.compile(filters["style"], re.I)}
#     if "fuel_type" in filters:
#         q["fuel_type"] = {"$regex": re.compile(filters["fuel_type"], re.I)}
#     if "query" in filters:
#         q["$or"] = [
#             {"make": {"$regex": re.compile(filters["query"], re.I)}},
#             {"model": {"$regex": re.compile(filters["query"], re.I)}},
#             {"description": {"$regex": re.compile(filters["query"], re.I)}}
#         ]
#     cursor = cars_col.find(q).sort([("year",-1),("price",1)]).limit(limit)
#     return [c for c in cursor]

# def tavily_search_raw(q: str, max_results: int = 3) -> List[Dict[str,Any]]:
#     if not TAVILY_API_KEY:
#         return [{"error":"TAVILY_API_KEY not configured"}]
#     try:
#         from tavily import TavilyClient
#         client = TavilyClient(TAVILY_API_KEY)
#         response = client.search(query=q, time_range="month")
#         results = response.get("results", [])[:max_results]
#         return results
#     except Exception as e:
#         return [{"error": f"Tavily request failed: {e}"}]

# # ---------- Tooling helpers ----------
# CAR_JSON_MARKER = "===CAR_JSON==="
# WEB_JSON_MARKER = "===WEB_JSON==="

# def extract_and_store_json_markers_safe(text: str, session_id: str, memory_manager: ConversationMemoryManager):
#     """Extract JSON from markers and store in session - improved version"""
#     if not text:
#         return

#     def _parse_json_after_marker(after: str):
#         """Try multiple strategies to parse JSON"""
#         s = after.lstrip()

#         # Strategy 1: Try standard JSON decoder
#         decoder = json.JSONDecoder()
#         for start_char in ('{', '['):
#             idx = s.find(start_char)
#             if idx != -1:
#                 try:
#                     obj, _ = decoder.raw_decode(s[idx:])
#                     return obj
#                 except Exception:
#                     pass

#         # Strategy 2: Regex extraction
#         try:
#             m = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', s, re.DOTALL)
#             if m:
#                 return json.loads(m.group(1))
#         except Exception:
#             pass

#         # Strategy 3: Find balanced braces/brackets
#         for start_char, end_char in [('{', '}'), ('[', ']')]:
#             idx = s.find(start_char)
#             if idx != -1:
#                 depth = 0
#                 for i, c in enumerate(s[idx:], idx):
#                     if c == start_char:
#                         depth += 1
#                     elif c == end_char:
#                         depth -= 1
#                         if depth == 0:
#                             try:
#                                 return json.loads(s[idx:i+1])
#                             except Exception:
#                                 break
#         return None

#     # Extract CAR_JSON
#     if CAR_JSON_MARKER in text:
#         try:
#             after = text.split(CAR_JSON_MARKER, 1)[1]
#             parsed = _parse_json_after_marker(after)
#             if parsed is not None:
#                 s = memory_manager.sessions.setdefault(session_id, memory_manager._new_session(""))
#                 s['last_results'] = parsed
#                 persist_session_state(session_id)
#             else:
#                 print(f"[extract CAR_JSON] Failed to parse. Snippet: {after[:200]}", file=sys.stderr)
#         except Exception as e:
#             print(f"[extract CAR_JSON error] {e}", file=sys.stderr)

#     # Extract WEB_JSON
#     if WEB_JSON_MARKER in text:
#         try:
#             after = text.split(WEB_JSON_MARKER, 1)[1]
#             parsed = _parse_json_after_marker(after)
#             if parsed is not None:
#                 s = memory_manager.sessions.setdefault(session_id, memory_manager._new_session(""))
#                 s['last_web_results'] = parsed
#                 persist_session_state(session_id)
#             else:
#                 print(f"[extract WEB_JSON] Failed to parse. Snippet: {after[:200]}", file=sys.stderr)
#         except Exception as e:
#             print(f"[extract WEB_JSON error] {e}", file=sys.stderr)

# def robust_extract_content(response) -> str:
#     """Extract clean text content from various response formats"""
#     if response is None:
#         return ""
#     if isinstance(response, str):
#         return response

#     try:
#         # Handle LangChain agent responses (dict with 'messages' key)
#         if isinstance(response, dict) and "messages" in response:
#             messages = response["messages"]
#             if isinstance(messages, list) and len(messages) > 0:
#                 # Find the last AIMessage with actual content (not tool_calls)
#                 for msg in reversed(messages):
#                     # Skip ToolMessages
#                     is_tool_msg = (
#                         (hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage') or
#                         (isinstance(msg, dict) and msg.get('type') == 'tool')
#                     )
#                     if is_tool_msg:
#                         continue

#                     # Check for AIMessage with content (and no tool_calls)
#                     has_tool_calls = False
#                     if hasattr(msg, 'tool_calls') and msg.tool_calls:
#                         has_tool_calls = True
#                     elif isinstance(msg, dict) and msg.get('tool_calls'):
#                         has_tool_calls = True

#                     # Extract content if it exists and no pending tool calls
#                     if not has_tool_calls:
#                         if hasattr(msg, 'content') and msg.content:
#                             return str(msg.content)
#                         if isinstance(msg, dict) and 'content' in msg and msg['content']:
#                             return str(msg['content'])

#         # Handle direct message objects with content attribute
#         if hasattr(response, "content"):
#             content = response.content
#             if isinstance(content, str) and content:
#                 return content
#             # Handle list of content blocks
#             if isinstance(content, list):
#                 text_parts = []
#                 for item in content:
#                     if isinstance(item, dict) and "text" in item:
#                         text_parts.append(item["text"])
#                     elif hasattr(item, "text"):
#                         text_parts.append(item.text)
#                     elif isinstance(item, str):
#                         text_parts.append(item)
#                 if text_parts:
#                     return "\n".join(text_parts)

#         # Handle OpenAI-style responses
#         if isinstance(response, dict):
#             if "choices" in response and len(response["choices"]) > 0:
#                 ch = response["choices"][0]
#                 if isinstance(ch, dict):
#                     if "message" in ch and "content" in ch["message"]:
#                         return ch["message"]["content"]
#                     if "text" in ch:
#                         return ch["text"]
#             # Handle direct content field
#             if "content" in response:
#                 content = response["content"]
#                 if isinstance(content, str) and content:
#                     return content

#         # Handle output attribute
#         if hasattr(response, "output"):
#             output = response.output
#             if isinstance(output, str):
#                 return output
#             # Recursively extract from output
#             return robust_extract_content(output)

#         # Last resort - convert to string but try to clean it
#         result = str(response)

#         # If it's a message dict dump, try to extract the last content
#         if ("'messages':" in result or '"messages":' in result) and len(result) > 200:
#             # Try to find the last content that's not empty
#             # Look for patterns like content='...' or content="..."
#             all_contents = re.findall(r"content=['\"]([^'\"]*(?:\\['\"][^'\"]*)*)['\"]", result)
#             if all_contents:
#                 # Return the last non-empty content
#                 for content in reversed(all_contents):
#                     if content and content.strip() and not content.startswith("HumanMessage"):
#                         return content.replace("\\'", "'").replace('\\"', '"')

#         return result
#     except Exception as e:
#         print(f"[robust_extract_content error] {e}", file=sys.stderr)
#         return str(response)

# def format_car_card(c: Dict[str, Any]) -> str:
#     """Format a single car as a readable line"""
#     if not c:
#         return "Unknown vehicle"
#     make = sanitize_text(str(c.get("make", "") or "Unknown"))
#     model = sanitize_text(str(c.get("model", "") or ""))
#     year = str(c.get("year", "") or "")
#     price = c.get("price")
#     if price is None or price == "":
#         price_str = "Price N/A"
#     else:
#         try:
#             if isinstance(price, (int, float)) and float(price).is_integer():
#                 price_str = f"${int(price):,}"
#             else:
#                 price_str = f"${float(price):,}"
#         except Exception:
#             price_str = str(price)
#     mileage = c.get("mileage")
#     mileage_str = f"{mileage} km" if mileage not in (None, "") else "Mileage N/A"
#     desc = sanitize_text(str(c.get("description", "") or ""))
#     if desc:
#         desc = desc.split(".")[0][:100]
#     title = " ".join(part for part in [make, model] if part).strip()
#     if year:
#         title = f"{title} ({year})"
#     return f"{title} — {price_str} — {mileage_str}" + (f" — {desc}" if desc else "")

# def build_results_message(cars: List[Dict[str, Any]]) -> str:
#     """Build human-readable car listing"""
#     if not cars:
#         return "No cars matched your filters."
#     total = len(cars)
#     top = cars[0] if total > 0 else None
#     lines = []
#     for i, c in enumerate(cars[:8], start=1):
#         lines.append(f"{i}. {format_car_card(c)}")
#     best_text = ""
#     if top:
#         make = top.get("make", "")
#         model = top.get("model", "")
#         year = top.get("year", "")
#         best_title = " ".join(part for part in [make, model] if part).strip()
#         if year:
#             best_title = f"{best_title} ({year})"
#         best_text = f"Top pick: {best_title}."
#     summary = f"I found {total} match{'es' if total != 1 else ''}. {best_text}\n"
#     summary += "Reply with the number to select a car, or say 'more filters' to narrow results."
#     return summary + "\n\n" + "\n".join(lines)

# # ---------- Tools ----------
# if LC_AVAILABLE:
#     @tool("get_user_profile", description="Fetch user profile (input: email). Returns profile text.")
#     def tool_get_user_profile(email: str) -> str:
#         return fetch_user_profile_by_email(email)

#     @tool("find_cars", description="Fetch cars in DB (input: JSON filters string). Returns formatted list with CAR_JSON marker.")
#     def tool_find_cars(filters_json: str) -> str:
#         try:
#             filters = json.loads(filters_json) if isinstance(filters_json, str) else {}
#         except Exception:
#             filters = {"query": filters_json}
#         cars = fetch_cars_by_filters(filters, limit=20)
#         out = []
#         for c in cars:
#             c2 = {k: v for k, v in c.items() if k != "_id"}
#             out.append(c2)

#         # Return CLEAN human text + JSON marker
#         if out:
#             human_text = build_results_message(out)
#             json_str = json.dumps(out, default=str)
#             return f"{human_text}\n\n{CAR_JSON_MARKER}{json_str}"
#         else:
#             return "I couldn't find any cars matching those filters. Try adjusting your criteria."

#     @tool("web_search", description="Search the web (input: query). Returns results with WEB_JSON marker.")
#     def tool_web_search(query: str) -> str:
#         results = tavily_search_raw(query, max_results=3)
#         human = "External search results:\n\n"
#         lines = []
#         for r in results:
#             if isinstance(r, dict) and r.get("error"):
#                 lines.append(r.get("error"))
#             else:
#                 title = r.get("title") or r.get("headline") or ""
#                 snippet = r.get("snippet") or r.get("summary") or ""
#                 url = r.get("url") or r.get("link") or ""
#                 lines.append(f"{title}\n{snippet}\n{url}")
#         human += "\n\n".join(lines) if lines else str(results)
#         return human + "\n\n" + WEB_JSON_MARKER + json.dumps(results, default=str)

#     @tool("place_order", description="Place order (input: JSON with session_id, buyer_address, etc.)")
#     def tool_place_order(payload: str) -> str:
#         try:
#             data = json.loads(payload)
#         except Exception:
#             return "Invalid payload. place_order expects JSON with session_id."
#         session_id = data.get("session_id")
#         if not session_id or session_id not in memory_manager.sessions:
#             return "Invalid or expired session_id. Please retry."
#         buyer_name = data.get("buyer_name")
#         vehicle = data.get("vehicle")
#         sales_contact = data.get("sales_contact")
#         buyer_address = data.get("buyer_address")
#         buyer_phone = data.get("buyer_phone")
#         buyer_email = data.get("buyer_email")

#         s = memory_manager.sessions.get(session_id)
#         if not vehicle and s:
#             vehicle = s.get("selected_vehicle")
#         try:
#             oid = create_order_with_address(
#                 session_id=session_id,
#                 buyer_name=buyer_name,
#                 vehicle=vehicle,
#                 sales_contact=sales_contact,
#                 buyer_address=buyer_address,
#                 buyer_phone=buyer_phone,
#                 buyer_email=buyer_email,
#             )
#             if oid:
#                 return f"✅ Order placed successfully! Order ID: {oid}"
#             else:
#                 return "Failed to place order (DB write returned None)."
#         except Exception as e:
#             print("[tool_place_order error]", e, file=sys.stderr)
#             try:
#                 failed_writes_col.insert_one({"collection": "orders", "error": str(e), "payload": _make_json_safe(data), "timestamp": utcnow_iso()})
#             except Exception:
#                 pass
#             return f"Error while placing order: {e}"

# # ---------- Create agents ----------
# personal_agent = None
# car_agent = None
# web_agent = None
# supervisor_agent = None

# if LC_AVAILABLE:
#     personal_prompt = (
#         "You are a Personal Agent. Fetch user profile when needed using get_user_profile tool.\n"
#         "Return clear, human-readable text."
#     )
#     personal_agent = create_agent(model=llm, name="PersonalAgent", system_prompt=personal_prompt, tools=[tool_get_user_profile])

#     car_prompt = (
#         "You are a Car Sales Agent. Search cars and place orders.\n"
#         "When returning results, use clean formatting followed by ===CAR_JSON=== marker with valid JSON.\n"
#         "Be helpful and conversational."
#     )
#     car_agent = create_agent(model=llm, name="CarSalesAgent", system_prompt=car_prompt, tools=[tool_find_cars, tool_place_order])

#     web_prompt = (
#         "You are a Web Agent. Search external sources using web_search tool when needed.\n"
#         "Return human-readable summaries."
#     )
#     web_agent = create_agent(model=llm, name="WebAgent", system_prompt=web_prompt, tools=[tool_web_search])

#     supervisor_system_prompt = (
#         "You are the Supervisor Agent coordinating sub-agents (Personal, Car, Web).\n"
#         "Rules:\n"
#         "1) Use personal_wrapper for user profile queries\n"
#         "2) Use car_wrapper for vehicle searches and orders\n"
#         "3) Use web_wrapper for external research\n"
#         "4) Extract JSON markers (===CAR_JSON===, ===WEB_JSON===) from tool responses\n"
#         "5) Return CLEAN, user-friendly text - no raw JSON or tool outputs\n"
#         "6) Be conversational and helpful"
#     )

#     @tool("personal_wrapper", description="Invoke Personal Agent")
#     def tool_personal_wrapper(payload: str) -> str:
#         if personal_agent:
#             try:
#                 resp = personal_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = personal_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Personal Agent error: {e}"
#         return "Personal Agent not available."

#     @tool("car_wrapper", description="Invoke Car Agent")
#     def tool_car_wrapper(payload: str) -> str:
#         if car_agent:
#             try:
#                 resp = car_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = car_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Car Agent error: {e}"
#         return "Car Agent not available."

#     @tool("web_wrapper", description="Invoke Web Agent")
#     def tool_web_wrapper(payload: str) -> str:
#         if web_agent:
#             try:
#                 resp = web_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = web_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Web Agent error: {e}"
#         return "Web Agent not available."

#     supervisor_agent = create_agent(model=llm, name="SupervisorAgent", system_prompt=supervisor_system_prompt, tools=[tool_personal_wrapper, tool_car_wrapper, tool_web_wrapper])

# # ---------- Helper functions ----------
# def is_order_confirmation(user_text: str) -> bool:
#     if not user_text:
#         return False
#     t = user_text.lower().strip()
#     keywords = ["confirm","place order","buy","purchase","i want this","proceed","yes i want","go ahead","yes"]
#     return any(k in t for k in keywords)

# def contains_address_info(user_text: str) -> bool:
#     """Check if user text contains address/contact information"""
#     if not user_text:
#         return False
#     t = user_text.lower()
#     address_indicators = ["address", "street", "road", "avenue", "city", "phone", "email", "name:"]
#     return any(indicator in t for indicator in address_indicators)

# def handle_car_selection(session_id: str, user_text: str) -> Optional[str]:
#     """Handle numeric car selection"""
#     s = memory_manager.sessions.get(session_id)
#     if not s:
#         return None
#     if not user_text or not user_text.strip().isdigit():
#         return None
#     if not s.get("last_results"):
#         return None
#     try:
#         idx = int(user_text.strip()) - 1
#         if idx < 0 or idx >= len(s["last_results"]):
#             return f"Selection {user_text.strip()} is out of range. Please choose between 1 and {len(s['last_results'])}."
#         sel = s["last_results"][idx]
#         try:
#             sel_copy = {k: (str(v) if k == "_id" else v) for k, v in sel.items()}
#         except Exception:
#             sel_copy = _make_json_safe(sel)
#         s["selected_vehicle"] = sel_copy
#         s["stage"] = "vehicle_selected"
#         s["awaiting"] = "address"  # Set awaiting status
#         persist_session_state(session_id)

#         response = (
#             f"✓ Great choice! You've selected:\n\n"
#             f"🚗 {sel_copy.get('make')} {sel_copy.get('model')} ({sel_copy.get('year')})\n"
#             f"💰 Price: ${sel_copy.get('price'):,}\n"
#             f"📏 Mileage: {sel_copy.get('mileage')} km\n\n"
#             f"To complete your order, please provide:\n"
#             f"• Your full name\n"
#             f"• Delivery address\n"
#             f"• Phone number\n"
#             f"• Email address\n\n"
#             f"You can provide them all at once or one at a time."
#         )
#         return response
#     except Exception as e:
#         print("[handle_car_selection]", e, file=sys.stderr)
#         return None

# # ---------- Main supervisor invoke ----------
# def supervisor_invoke(session_id: str, user_email: str, user_query: str) -> Tuple[str, str]:
#     """Main orchestrator with memory optimization and clean output"""
#     session = memory_manager.sessions.get(session_id, memory_manager._new_session(user_email))

#     # Extract contact info if present
#     contact_info = extract_contact_info(user_query)
#     if contact_info:
#         print(f"[supervisor_invoke] Extracted contact info: {contact_info}", file=sys.stderr)
#         collected = session.get("collected", {})
#         collected.update(contact_info)
#         session["collected"] = collected
#         persist_session_state(session_id)

#     # Check if we're awaiting address and user provided it
#     awaiting = session.get("awaiting")
#     selected_vehicle = session.get("selected_vehicle")
#     already_ordered = bool(session.get("order_id"))

#     if awaiting == "address" and contains_address_info(user_query) and selected_vehicle and not already_ordered:
#         print(f"[supervisor_invoke] Address provided, attempting order placement", file=sys.stderr)
#         collected = session.get("collected", {})
#         buyer_address = collected.get("address")

#         if buyer_address:
#             try:
#                 oid = create_order_with_address(
#                     session_id=session_id,
#                     buyer_name=collected.get("name") or user_email,
#                     vehicle=selected_vehicle,
#                     sales_contact={
#                         "name": "Jeni Flemin",
#                         "position": "CEO",
#                         "phone": "+94778540035",
#                         "address": "Convent Garden, London, UK"
#                     },
#                     buyer_address=buyer_address,
#                     buyer_phone=collected.get("phone"),
#                     buyer_email=collected.get("email") or user_email
#                 )

#                 if oid:
#                     session["awaiting"] = None
#                     persist_session_state(session_id)

#                     success_msg = (
#                         f"✅ Perfect! Your order has been placed successfully.\n\n"
#                         f"📋 Order Details:\n"
#                         f"• Order ID: {oid}\n"
#                         f"• Vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})\n"
#                         f"• Price: ${selected_vehicle.get('price'):,}\n"
#                         f"• Delivery to: {buyer_address}\n\n"
#                         f"Our sales team will contact you at {collected.get('phone', 'your provided number')} within 24 hours to confirm delivery details.\n\n"
#                         f"Is there anything else I can help you with?"
#                     )
#                     memory_manager.add_message(session_id, user_query, success_msg, agent_used="OrderHandler")
#                     return success_msg, session_id

#             except Exception as e:
#                 error_msg = f"I apologize, but there was an error placing your order: {str(e)}. Please try again or contact our support team."
#                 print(f"[supervisor_invoke] Order placement error: {e}", file=sys.stderr)
#                 memory_manager.add_message(session_id, user_query, error_msg, agent_used="OrderHandler")
#                 return error_msg, session_id

#     # Handle numeric selection first
#     sel_reply = handle_car_selection(session_id, user_query)
#     if sel_reply is not None:
#         memory_manager.add_message(session_id, user_query, sel_reply, agent_used="SelectionHandler")
#         return sel_reply, session_id

#     # Build context with memory optimization
#     conversation_context = memory_manager.get_context_for_llm(session_id) if hasattr(memory_manager, 'get_context_for_llm') else None

#     # Add session state context
#     state_context = ""
#     if selected_vehicle and not already_ordered:
#         state_context = f"\n[Current selected vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})]"
#         if awaiting == "address":
#             state_context += "\n[Awaiting delivery address from user]"

#     full_prompt = f"\nPrevious conversation:\n{conversation_context}\n{state_context}\n\nUser: {user_query}\n" if conversation_context else user_query

#     # Get response from supervisor - run agent to completion
#     if LC_AVAILABLE and supervisor_agent:
#         try:
#             # Run agent with proper message format
#             messages = [{"role": "user", "content": full_prompt}]
#             resp = supervisor_agent.invoke({"messages": messages})

#             # Keep invoking until we get a final response (no tool_calls in last message)
#             max_iterations = 10
#             iteration = 0

#             while iteration < max_iterations:
#                 iteration += 1

#                 # Check if response has messages
#                 if isinstance(resp, dict) and "messages" in resp:
#                     messages_list = resp["messages"]
#                     if not messages_list:
#                         break

#                     last_msg = messages_list[-1]

#                     # Check if last message has tool_calls (needs to continue)
#                     has_tool_calls = False
#                     if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                         has_tool_calls = True
#                     elif isinstance(last_msg, dict) and last_msg.get('tool_calls'):
#                         has_tool_calls = True

#                     # If the last message is a ToolMessage, we need another iteration for AI to respond
#                     is_tool_message = (
#                         (hasattr(last_msg, '__class__') and last_msg.__class__.__name__ == 'ToolMessage') or
#                         (isinstance(last_msg, dict) and last_msg.get('type') == 'tool')
#                     )

#                     # Continue if we have tool calls OR if last message is a tool result
#                     if has_tool_calls or is_tool_message:
#                         try:
#                             resp = supervisor_agent.invoke(resp)
#                         except Exception as e:
#                             print(f"[agent continuation error] {e}", file=sys.stderr)
#                             break
#                     else:
#                         # Last message is AIMessage with content, we're done
#                         break
#                 else:
#                     # Not in expected format, break
#                     break

#             out_raw = robust_extract_content(resp)

#         except Exception as e:
#             print(f"[supervisor error] {e}", file=sys.stderr)
#             import traceback
#             traceback.print_exc(file=sys.stderr)
#             out_raw = f"Sorry, I encountered an error: {e}"
#     else:
#         try:
#             resp = llm([{"role": "user", "content": full_prompt}])
#             out_raw = robust_extract_content(resp)
#         except Exception as e:
#             out_raw = f"Fallback response: {user_query} (error: {e})"

#     # Extract JSON markers into session
#     try:
#         extract_and_store_json_markers_safe(str(out_raw), session_id, memory_manager)
#     except Exception as e:
#         print("[extract_and_store_json_markers error]", e, file=sys.stderr)

#     # Clean the output - remove JSON markers from user-facing text
#     cleaned_output = out_raw
#     if CAR_JSON_MARKER in cleaned_output:
#         cleaned_output = cleaned_output.split(CAR_JSON_MARKER)[0]
#     if WEB_JSON_MARKER in cleaned_output:
#         cleaned_output = cleaned_output.split(WEB_JSON_MARKER)[0]
#     cleaned_output = cleaned_output.strip()

#     # Save to conversation history
#     memory_manager.add_message(session_id, user_query, cleaned_output, agent_used="Supervisor")

#     return cleaned_output, session_id

# # ---------- Top-level API ----------
# def handle_user_query(session_id: Optional[str], user_email: str, user_query: str) -> Dict[str,Any]:
#     """Main entry point for handling user queries"""
#     sid = memory_manager.get_or_create_session(user_email, session_id)
#     memory_manager.sessions[sid]["user_email"] = user_email
#     resp, _ = supervisor_invoke(sid, user_email, user_query)
#     return {"response": resp, "session_id": sid}

# def end_session(session_id: str) -> str:
#     """End session and generate summary"""
#     return memory_manager.end_session_and_save(session_id)

# __all__ = [
#     'handle_user_query',
#     'end_session',
#     'memory_manager',
#     'create_order_with_address',
#     'persist_session_state'
# ]


"""
Final agent.py - Fixed version with proper order placement
"""

# import os
# import re
# import json
# import time
# import sys
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime, timezone, date
# from dotenv import load_dotenv

# load_dotenv()

# MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# if not MONGO_URI:
#     raise RuntimeError("MONGODB_CONNECTION_STRING not set in .env")

# # ---------- MongoDB ----------
# from pymongo import MongoClient
# mongo = MongoClient(MONGO_URI)
# db = mongo.get_database()
# cars_col = db.get_collection("cars")
# users_col = db.get_collection("users")
# convos_col = db.get_collection("conversations")
# summaries_col = db.get_collection("conversation_summaries")
# orders_col = db.get_collection("orders")
# failed_writes_col = db.get_collection("failed_writes")

# # ---------- LangGraph Memory ----------
# try:
#     from langgraph.checkpoint.memory import InMemorySaver
#     from langgraph.store.memory import InMemoryStore
#     from langchain_core.messages import HumanMessage, AIMessage
#     LANGGRAPH_AVAILABLE = True
# except Exception:
#     LANGGRAPH_AVAILABLE = False
#     InMemorySaver = InMemoryStore = None
#     class HumanMessage:
#         def __init__(self, content: str):
#             self.content = content
#     class AIMessage:
#         def __init__(self, content: str):
#             self.content = content

# # ---------- LangChain / OpenAI detection ----------
# llm = None
# LC_AVAILABLE = False
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain.agents import create_agent
#     from langchain.tools import tool
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
#     LC_AVAILABLE = True
# except Exception:
#     import openai
#     openai.api_key = OPENAI_API_KEY
#     class SimpleOpenAIWrapper:
#         def __init__(self, model=LLM_MODEL_NAME, temperature=0):
#             self.model = model
#             self.temperature = temperature
#         def __call__(self, messages: List[Dict[str,str]]):
#             return openai.ChatCompletion.create(model=self.model, messages=messages, temperature=self.temperature)
#     llm = SimpleOpenAIWrapper()
#     LC_AVAILABLE = False

# # ---------- Utilities ----------
# def utcnow_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def sanitize_text(s: str, max_len: int = 4000) -> str:
#     if s is None:
#         return ""
#     if not isinstance(s, str):
#         s = str(s)
#     s = re.sub(r"\s+", " ", s).strip()
#     if len(s) > max_len:
#         return s[:max_len] + "..."
#     return s

# def _make_json_safe(obj: Any) -> Any:
#     try:
#         return json.loads(json.dumps(obj, default=str))
#     except Exception:
#         return str(obj)

# def normalize_vehicle(vehicle):
#     """Ensure vehicle is a dict."""
#     if not vehicle:
#         return None
#     if isinstance(vehicle, dict):
#         return vehicle
#     return None

# def estimate_tokens(text: str) -> int:
#     if not text:
#         return 0
#     return max(1, int(len(text) / 4))

# def extract_contact_info(text: str) -> Dict[str, str]:
#     """Extract contact information from user input"""
#     info = {}

#     # Extract name
#     name_match = re.search(r'name\s*:?\s*([^,\n]+)', text, re.IGNORECASE)
#     if name_match:
#         info['name'] = name_match.group(1).strip()

#     # Extract phone
#     phone_match = re.search(r'phone\s*:?\s*([\+\d\s\-\(\)]+)', text, re.IGNORECASE)
#     if phone_match:
#         info['phone'] = phone_match.group(1).strip()

#     # Extract email
#     email_match = re.search(r'email\s*:?\s*([^\s,]+@[^\s,]+)', text, re.IGNORECASE)
#     if email_match:
#         info['email'] = email_match.group(1).strip()

#     # Extract address (look for address: or delivery address:)
#     address_match = re.search(r'(?:delivery\s+)?address\s*:?\s*([^,]+(?:,[^,]+)*)', text, re.IGNORECASE)
#     if address_match:
#         info['address'] = address_match.group(1).strip()

#     return info

# # ---------- Memory optimizer mixin ----------
# class MemoryOptimizerMixin:
#     MAX_PROMPT_TOKENS = 3000
#     RECENT_TURNS_KEEP = 8
#     SUMMARIZE_EVERY = 12
#     SUMMARY_MAX_TOKENS = 800

#     def compress_history_if_needed(self, session_id: str):
#         s = self.sessions.get(session_id)
#         if not s:
#             return
#         msgs = s.get("messages", [])
#         if len(msgs) <= (self.RECENT_TURNS_KEEP + 2):
#             return
#         last_summary_at = s.get("_last_summary_index", 0)
#         if len(msgs) - last_summary_at < self.SUMMARIZE_EVERY:
#             return
#         older = msgs[: max(0, len(msgs) - self.RECENT_TURNS_KEEP)]
#         if not older:
#             return
#         older_text = []
#         for m in older:
#             u = m.get("user") or ""
#             a = m.get("assistant") or ""
#             if u:
#                 older_text.append(f"User: {sanitize_text(u, 2000)}")
#             if a:
#                 older_text.append(f"Assistant: {sanitize_text(a, 2000)}")
#         to_summarize = "\n".join(older_text)
#         if estimate_tokens(to_summarize) < (self.SUMMARY_MAX_TOKENS // 2):
#             summary = to_summarize
#         else:
#             prompt = ("Summarize the following conversation history into concise bullet points. "
#                       "Keep facts, decisions, selected vehicle details, outstanding questions and next steps. "
#                       "Limit to ~200-400 words.\n\nHistory:\n" + to_summarize + "\n\nSummary:")
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#                 if not summary:
#                     summary = "(summary generation failed)"
#             except Exception as e:
#                 summary = f"(summary generation failed: {e})"
#         prev = s.get('memory_summary', '') or ''
#         new_summary = (prev + '\n---\n' + summary) if prev else summary
#         recent = msgs[-self.RECENT_TURNS_KEEP:]
#         placeholder = {"user": "[older history summarized]", "assistant": new_summary, "agent": "system_summary", "timestamp": utcnow_iso()}
#         s['messages'] = [placeholder] + recent
#         s['_last_summary_index'] = len(s['messages'])
#         s['memory_summary'] = new_summary
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[compress_history persist error]", e, file=sys.stderr)

#     def get_context_for_llm(self, session_id: str, max_messages: int = None) -> str:
#         s = self.sessions.get(session_id)
#         if not s:
#             return ""
#         try:
#             self.compress_history_if_needed(session_id)
#         except Exception:
#             pass
#         memory_summary = s.get('memory_summary', '') or ''
#         recent = s.get('messages', [])[-self.RECENT_TURNS_KEEP:]
#         lines = []
#         tokens_used = 0
#         if memory_summary:
#             ts = f"Memory Summary:\n{memory_summary}\n"
#             t_count = estimate_tokens(ts)
#             lines.append(ts)
#             tokens_used += t_count
#         for m in recent:
#             u = m.get('user') or ''
#             a = m.get('assistant') or ''
#             agent = m.get('agent') or ''
#             if u:
#                 line = f"User: {u}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#             if a:
#                 line = f"Assistant ({agent}): {a}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#         return "\n".join(lines)

# # ---------- Conversation memory manager ----------
# class ConversationMemoryManager(MemoryOptimizerMixin):
#     def __init__(self):
#         super().__init__()
#         self.sessions: Dict[str, Dict[str, Any]] = {}
#         if LANGGRAPH_AVAILABLE:
#             try:
#                 self.checkpointer = InMemorySaver()
#                 self.store = InMemoryStore()
#             except Exception:
#                 self.checkpointer = None
#                 self.store = None
#         else:
#             self.checkpointer = None
#             self.store = None

#     def _new_session(self, user_email: str) -> Dict[str,Any]:
#         return {
#             "user_email": user_email,
#             "start_time": utcnow_iso(),
#             "messages": [],
#             "stage": "init",
#             "collected": {},
#             "last_results": [],
#             "last_web_results": [],
#             "selected_vehicle": None,
#             "order_id": None,
#             "memory_summary": "",
#             "awaiting": None
#         }

#     def hydrate_langgraph_memory(self, session_id: str):
#         if not self.checkpointer:
#             return
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1).limit(50))
#             msgs = []
#             for r in rows:
#                 u = r.get("user_message")
#                 b = r.get("bot_response")
#                 if u:
#                     msgs.append(HumanMessage(content=u))
#                 if b:
#                     msgs.append(AIMessage(content=b))
#             if msgs:
#                 try:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs})
#                 except TypeError:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs}, {})
#         except Exception:
#             pass

#     def get_or_create_session(self, user_email: str, session_id: Optional[str] = None) -> str:
#         if session_id:
#             if session_id not in self.sessions:
#                 self.sessions[session_id] = self._new_session(user_email)
#                 try:
#                     u = users_col.find_one({"email": user_email})
#                     if u and "current_session" in u and u["current_session"].get("session_id") == session_id:
#                         cs = u["current_session"]
#                         s = self.sessions[session_id]
#                         s["stage"] = cs.get("stage", s["stage"])
#                         s["selected_vehicle"] = cs.get("selected_vehicle", s["selected_vehicle"])
#                         s["order_id"] = cs.get("order_id", s["order_id"])
#                         s["collected"] = cs.get("collected", s["collected"])
#                 except Exception:
#                     pass
#                 try:
#                     self.hydrate_langgraph_memory(session_id)
#                 except Exception:
#                     pass
#             return session_id
#         sid = f"{user_email}_{int(time.time())}"
#         self.sessions[sid] = self._new_session(user_email)
#         persist_session_state_raw(user_email, sid, self.sessions[sid])
#         return sid

#     def add_message(self, session_id: str, user_message: str, bot_response: str, agent_used: str):
#         if session_id not in self.sessions:
#             self.sessions[session_id] = self._new_session("")
#         user_message = sanitize_text(user_message, max_len=4000)
#         bot_response = sanitize_text(bot_response, max_len=4000)
#         entry = {"user": user_message, "assistant": bot_response, "agent": agent_used, "timestamp": utcnow_iso()}
#         self.sessions[session_id]["messages"].append(entry)
#         try:
#             conv_doc = {
#                 "session_id": session_id,
#                 "user_email": self.sessions[session_id].get("user_email", ""),
#                 "user_message": user_message,
#                 "bot_response": bot_response,
#                 "agent_used": agent_used,
#                 "timestamp": utcnow_iso(),
#                 "turn_index": len(self.sessions[session_id]["messages"]) - 1
#             }
#             conv_doc_safe = _make_json_safe(conv_doc)
#             convos_col.insert_one(conv_doc_safe)
#         except Exception as e:
#             print("[convos_col insert error]", e, file=sys.stderr)
#             try:
#                 failed_writes_col.insert_one({"collection": "conversations", "error": str(e), "doc": _make_json_safe(conv_doc), "timestamp": utcnow_iso()})
#             except Exception:
#                 pass
#         if self.checkpointer:
#             try:
#                 config = {"configurable": {"thread_id": session_id}}
#                 state = {"messages": [HumanMessage(content=user_message), AIMessage(content=bot_response)]}
#                 try:
#                     self.checkpointer.put(config, state, {})
#                 except TypeError:
#                     self.checkpointer.put(config, state)
#             except Exception:
#                 pass
#         if self.store:
#             try:
#                 namespace = ("conversations", session_id)
#                 key = f"msg_{len(self.sessions[session_id]['messages'])}"
#                 try:
#                     self.store.put(namespace, key, entry)
#                 except TypeError:
#                     self.store.put(namespace, key, entry, {})
#             except Exception:
#                 pass
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[persist_session_state error]", e, file=sys.stderr)

#     def get_session_messages(self, session_id: str) -> List[Dict[str,Any]]:
#         if session_id in self.sessions and self.sessions[session_id]["messages"]:
#             return self.sessions[session_id]["messages"]
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1))
#             if rows:
#                 out = []
#                 for r in rows:
#                     out.append({
#                         "user": r.get("user_message"),
#                         "assistant": r.get("bot_response"),
#                         "agent": r.get("agent_used"),
#                         "timestamp": r.get("timestamp")
#                     })
#                 if session_id not in self.sessions:
#                     self.sessions[session_id] = self._new_session(rows[0].get("user_email",""))
#                 self.sessions[session_id]["messages"] = out
#                 return out
#         except Exception as e:
#             print("[get_session_messages error]", e, file=sys.stderr)
#         if self.store is not None:
#             try:
#                 namespace = ("conversations", session_id)
#                 items = self.store.search(namespace)
#                 if items:
#                     return [it.value for it in items]
#             except Exception:
#                 pass
#         return []

#     def generate_summary(self, session_id: str) -> str:
#         msgs = self.get_session_messages(session_id)
#         if not msgs:
#             return "No messages to summarize."
#         convo_text = []
#         for m in msgs:
#             convo_text.append(f"User: {m.get('user')}")
#             convo_text.append(f"Assistant: {m.get('assistant')}")
#         prompt = ("Summarize the following conversation concisely. Include main topics, the selected vehicle (if chosen), and next steps.\n\n"
#                   "Conversation:\n" + "\n".join(convo_text) + "\n\nSummary:")
#         if llm:
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#             except Exception:
#                 summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         else:
#             summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         return sanitize_text(summary, max_len=1000)

#     def end_session_and_save(self, session_id: str):
#         if session_id not in self.sessions:
#             return "No session messages to summarize."
#         summary = self.generate_summary(session_id)
#         msgs = self.sessions[session_id]["messages"]
#         message_count = len(msgs)
#         start_time = self.sessions[session_id].get("start_time")
#         end_time = utcnow_iso()
#         user_email = self.sessions[session_id].get("user_email","")
#         try:
#             summaries_col.update_one(
#                 {"session_id": session_id},
#                 {"$set": {
#                     "session_id": session_id,
#                     "user_email": user_email,
#                     "summary": summary,
#                     "message_count": message_count,
#                     "start_time": start_time,
#                     "end_time": end_time,
#                     "created_at": utcnow_iso()
#                 }}, upsert=True
#             )
#             if user_email:
#                 users_col.update_one({"email": user_email},
#                                      {"$set": {"recent_summary": summary, "last_session_id": session_id}},
#                                      upsert=True)
#         except Exception as e:
#             print("[end_session_and_save error]", e, file=sys.stderr)
#         self.sessions[session_id]["stage"] = "finished"
#         try:
#             persist_session_state(session_id)
#         except Exception:
#             pass
#         return summary

# # ---------- Order helpers ----------
# def create_order_with_address(
#     session_id: str,
#     buyer_name: Optional[str] = None,
#     vehicle: Optional[Dict[str, Any]] = None,
#     sales_contact: Optional[Dict[str, Any]] = None,
#     buyer_address: Optional[str] = None,
#     buyer_phone: Optional[str] = None,
#     buyer_email: Optional[str] = None,
# ) -> Optional[str]:
#     session = memory_manager.sessions.get(session_id)
#     if not session:
#         raise ValueError("Invalid session_id")

#     # Get buyer details from collected info
#     collected = session.get("collected", {})
#     if not buyer_name:
#         buyer_name = collected.get("name") or session.get("user_email")
#     if not buyer_address:
#         buyer_address = collected.get("address")
#     if not buyer_phone:
#         buyer_phone = collected.get("phone")
#     if not buyer_email:
#         buyer_email = collected.get("email") or session.get("user_email")

#     vehicle = normalize_vehicle(vehicle) or normalize_vehicle(session.get("selected_vehicle"))

#     if not vehicle:
#         raise ValueError("No vehicle selected for order")
#     if not buyer_address:
#         raise ValueError("Buyer address is required to place order")
#     if not isinstance(vehicle, dict):
#         raise ValueError("Selected vehicle data is invalid.")

#     order_doc = {
#         "session_id": session_id,
#         "user_email": session.get("user_email"),
#         "buyer_name": buyer_name,
#         "buyer_address": buyer_address,
#         "buyer_phone": buyer_phone,
#         "buyer_email": buyer_email,
#         "vehicle": {
#             "make": vehicle.get("make"),
#             "model": vehicle.get("model"),
#             "year": vehicle.get("year"),
#             "price": vehicle.get("price"),
#             "mileage": vehicle.get("mileage"),
#         },
#         "sales_contact": sales_contact or {
#             "name": "Jeni Flemin",
#             "position": "CEO",
#             "phone": "+94778540035",
#             "address": "Convent Garden, London, UK",
#         },
#         "timestamp": utcnow_iso(),
#         "order_date": date.today().isoformat(),
#         "conversation_summary": session.get("memory_summary", ""),
#     }

#     print(f"[create_order] Attempting to insert order: {json.dumps(order_doc, default=str, indent=2)}", file=sys.stderr)

#     try:
#         result = orders_col.insert_one(_make_json_safe(order_doc))
#         order_id = str(result.inserted_id)
#         session["order_id"] = order_id
#         session["stage"] = "ordered"
#         persist_session_state(session_id)
#         print(f"[create_order] Order created successfully with ID: {order_id}", file=sys.stderr)
#         return order_id
#     except Exception as e:
#         print(f"[create_order] Failed to insert order: {e}", file=sys.stderr)
#         try:
#             failed_writes_col.insert_one({
#                 "collection": "orders",
#                 "error": str(e),
#                 "doc": _make_json_safe(order_doc),
#                 "timestamp": utcnow_iso()
#             })
#         except Exception as e2:
#             print(f"[create_order] Failed to log to failed_writes: {e2}", file=sys.stderr)
#         raise

# memory_manager = ConversationMemoryManager()

# # ---------- Helpers ----------
# def persist_session_state(session_id: str):
#     s = memory_manager.sessions.get(session_id)
#     if not s:
#         return
#     email = s.get("user_email", "")
#     try:
#         users_col.update_one(
#             {"email": email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": s.get("stage"),
#                     "selected_vehicle": _make_json_safe(s.get("selected_vehicle")),
#                     "order_id": s.get("order_id"),
#                     "memory_summary": s.get("memory_summary", ""),
#                     "collected": _make_json_safe(s.get("collected", {})),
#                     "awaiting": s.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state]", e, file=sys.stderr)

# def persist_session_state_raw(user_email: str, session_id: str, session_obj: Dict[str,Any]):
#     try:
#         users_col.update_one(
#             {"email": user_email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": session_obj.get("stage"),
#                     "selected_vehicle": _make_json_safe(session_obj.get("selected_vehicle")),
#                     "order_id": session_obj.get("order_id"),
#                     "memory_summary": session_obj.get("memory_summary", ""),
#                     "collected": _make_json_safe(session_obj.get("collected", {})),
#                     "awaiting": session_obj.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": user_email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state_raw]", e, file=sys.stderr)

# def fetch_user_profile_by_email(email: str) -> str:
#     if not email:
#         return "No email provided."
#     p = users_col.find_one({"email": email})
#     if not p:
#         return f"No profile found for {email}."
#     return f"Name: {p.get('name','')}\nEmail: {p.get('email','')}\nRecent summary: {p.get('recent_summary')}"


# import os
# import re
# import json
# import time
# import sys
# from typing import List, Dict, Any, Optional, Tuple
# from datetime import datetime, timezone, date
# from dotenv import load_dotenv

# load_dotenv()

# MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# if not MONGO_URI:
#     raise RuntimeError("MONGODB_CONNECTION_STRING not set in .env")

# # ---------- MongoDB ----------
# from pymongo import MongoClient
# mongo = MongoClient(MONGO_URI)
# db = mongo.get_database()
# cars_col = db.get_collection("cars")
# users_col = db.get_collection("users")
# convos_col = db.get_collection("conversations")
# summaries_col = db.get_collection("conversation_summaries")
# orders_col = db.get_collection("orders")
# failed_writes_col = db.get_collection("failed_writes")

# # ---------- LangGraph Memory ----------
# try:
#     from langgraph.checkpoint.memory import InMemorySaver
#     from langgraph.store.memory import InMemoryStore
#     from langchain_core.messages import HumanMessage, AIMessage
#     LANGGRAPH_AVAILABLE = True
# except Exception:
#     LANGGRAPH_AVAILABLE = False
#     InMemorySaver = InMemoryStore = None
#     class HumanMessage:
#         def __init__(self, content: str):
#             self.content = content
#     class AIMessage:
#         def __init__(self, content: str):
#             self.content = content

# # ---------- LangChain / OpenAI detection ----------
# llm = None
# LC_AVAILABLE = False
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain.agents import create_agent
#     from langchain.tools import tool
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
#     LC_AVAILABLE = True
# except Exception:
#     import openai
#     openai.api_key = OPENAI_API_KEY
#     class SimpleOpenAIWrapper:
#         def __init__(self, model=LLM_MODEL_NAME, temperature=0):
#             self.model = model
#             self.temperature = temperature
#         def __call__(self, messages: List[Dict[str,str]]):
#             return openai.ChatCompletion.create(model=self.model, messages=messages, temperature=self.temperature)
#     llm = SimpleOpenAIWrapper()
#     LC_AVAILABLE = False

# # ---------- Utilities ----------
# def utcnow_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def sanitize_text(s: str, max_len: int = 4000) -> str:
#     if s is None:
#         return ""
#     if not isinstance(s, str):
#         s = str(s)
#     s = re.sub(r"\s+", " ", s).strip()
#     if len(s) > max_len:
#         return s[:max_len] + "..."
#     return s

# def _make_json_safe(obj: Any) -> Any:
#     try:
#         return json.loads(json.dumps(obj, default=str))
#     except Exception:
#         return str(obj)

# def normalize_vehicle(vehicle):
#     """Ensure vehicle is a dict."""
#     if not vehicle:
#         return None
#     if isinstance(vehicle, dict):
#         return vehicle
#     return None

# def estimate_tokens(text: str) -> int:
#     if not text:
#         return 0
#     return max(1, int(len(text) / 4))

# def extract_contact_info(text: str) -> Dict[str, str]:
#     """Extract contact information from user input"""
#     info = {}

#     # Extract name
#     name_match = re.search(r'name\s*:?\s*([^,\n]+)', text, re.IGNORECASE)
#     if name_match:
#         info['name'] = name_match.group(1).strip()

#     # Extract phone
#     phone_match = re.search(r'phone\s*:?\s*([\+\d\s\-\(\)]+)', text, re.IGNORECASE)
#     if phone_match:
#         info['phone'] = phone_match.group(1).strip()

#     # Extract email
#     email_match = re.search(r'email\s*:?\s*([^\s,]+@[^\s,]+)', text, re.IGNORECASE)
#     if email_match:
#         info['email'] = email_match.group(1).strip()

#     # Extract address (look for address: or delivery address:)
#     address_match = re.search(r'(?:delivery\s+)?address\s*:?\s*([^,]+(?:,[^,]+)*)', text, re.IGNORECASE)
#     if address_match:
#         info['address'] = address_match.group(1).strip()

#     return info

# # ---------- Memory optimizer mixin ----------
# class MemoryOptimizerMixin:
#     MAX_PROMPT_TOKENS = 3000
#     RECENT_TURNS_KEEP = 8
#     SUMMARIZE_EVERY = 12
#     SUMMARY_MAX_TOKENS = 800

#     def compress_history_if_needed(self, session_id: str):
#         s = self.sessions.get(session_id)
#         if not s:
#             return
#         msgs = s.get("messages", [])
#         if len(msgs) <= (self.RECENT_TURNS_KEEP + 2):
#             return
#         last_summary_at = s.get("_last_summary_index", 0)
#         if len(msgs) - last_summary_at < self.SUMMARIZE_EVERY:
#             return
#         older = msgs[: max(0, len(msgs) - self.RECENT_TURNS_KEEP)]
#         if not older:
#             return
#         older_text = []
#         for m in older:
#             u = m.get("user") or ""
#             a = m.get("assistant") or ""
#             if u:
#                 older_text.append(f"User: {sanitize_text(u, 2000)}")
#             if a:
#                 older_text.append(f"Assistant: {sanitize_text(a, 2000)}")
#         to_summarize = "\n".join(older_text)
#         if estimate_tokens(to_summarize) < (self.SUMMARY_MAX_TOKENS // 2):
#             summary = to_summarize
#         else:
#             prompt = ("Summarize the following conversation history into concise bullet points. "
#                       "Keep facts, decisions, selected vehicle details, outstanding questions and next steps. "
#                       "Limit to ~200-400 words.\n\nHistory:\n" + to_summarize + "\n\nSummary:")
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#                 if not summary:
#                     summary = "(summary generation failed)"
#             except Exception as e:
#                 summary = f"(summary generation failed: {e})"
#         prev = s.get('memory_summary', '') or ''
#         new_summary = (prev + '\n---\n' + summary) if prev else summary
#         recent = msgs[-self.RECENT_TURNS_KEEP:]
#         placeholder = {"user": "[older history summarized]", "assistant": new_summary, "agent": "system_summary", "timestamp": utcnow_iso()}
#         s['messages'] = [placeholder] + recent
#         s['_last_summary_index'] = len(s['messages'])
#         s['memory_summary'] = new_summary
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[compress_history persist error]", e, file=sys.stderr)

#     def get_context_for_llm(self, session_id: str, max_messages: int = None) -> str:
#         s = self.sessions.get(session_id)
#         if not s:
#             return ""
#         try:
#             self.compress_history_if_needed(session_id)
#         except Exception:
#             pass
#         memory_summary = s.get('memory_summary', '') or ''
#         recent = s.get('messages', [])[-self.RECENT_TURNS_KEEP:]
#         lines = []
#         tokens_used = 0
#         if memory_summary:
#             ts = f"Memory Summary:\n{memory_summary}\n"
#             t_count = estimate_tokens(ts)
#             lines.append(ts)
#             tokens_used += t_count
#         for m in recent:
#             u = m.get('user') or ''
#             a = m.get('assistant') or ''
#             agent = m.get('agent') or ''
#             if u:
#                 line = f"User: {u}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#             if a:
#                 line = f"Assistant ({agent}): {a}\n"
#                 t = estimate_tokens(line)
#                 if tokens_used + t > self.MAX_PROMPT_TOKENS:
#                     break
#                 lines.append(line)
#                 tokens_used += t
#         return "\n".join(lines)

# # ---------- Conversation memory manager ----------
# class ConversationMemoryManager(MemoryOptimizerMixin):
#     def __init__(self):
#         super().__init__()
#         self.sessions: Dict[str, Dict[str, Any]] = {}
#         if LANGGRAPH_AVAILABLE:
#             try:
#                 self.checkpointer = InMemorySaver()
#                 self.store = InMemoryStore()
#             except Exception:
#                 self.checkpointer = None
#                 self.store = None
#         else:
#             self.checkpointer = None
#             self.store = None

#     def _new_session(self, user_email: str) -> Dict[str,Any]:
#         return {
#             "user_email": user_email,
#             "start_time": utcnow_iso(),
#             "messages": [],
#             "stage": "init",
#             "collected": {},
#             "last_results": [],
#             "last_web_results": [],
#             "selected_vehicle": None,
#             "order_id": None,
#             "memory_summary": "",
#             "awaiting": None
#         }

#     def hydrate_langgraph_memory(self, session_id: str):
#         if not self.checkpointer:
#             return
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1).limit(50))
#             msgs = []
#             for r in rows:
#                 u = r.get("user_message")
#                 b = r.get("bot_response")
#                 if u:
#                     msgs.append(HumanMessage(content=u))
#                 if b:
#                     msgs.append(AIMessage(content=b))
#             if msgs:
#                 try:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs})
#                 except TypeError:
#                     self.checkpointer.put({"configurable": {"thread_id": session_id}}, {"messages": msgs}, {})
#         except Exception:
#             pass

#     def ensure_session_loaded(self, session_id: str, user_email: str = "") -> bool:
#         """
#         Ensure session_id exists in self.sessions. If not, attempt to rebuild it from users/current_session
#         and recent conversation rows (convos_col). Returns True if session loaded or already present.
#         This is intentionally robust: it tries multiple DB lookups and falls back to recent conv rows by user_email.
#         """
#         if not session_id:
#             return False
#         # Already loaded
#         if session_id in self.sessions:
#             return True
#         try:
#             # Direct match in users.current_session
#             u = users_col.find_one({"current_session.session_id": session_id})
#             if u:
#                 s = self._new_session(u.get("email") or user_email)
#                 cs = u.get("current_session", {})
#                 s["stage"] = cs.get("stage", s["stage"])
#                 s["selected_vehicle"] = cs.get("selected_vehicle", s["selected_vehicle"])
#                 s["order_id"] = cs.get("order_id", s["order_id"])
#                 s["collected"] = cs.get("collected", s["collected"])
#                 s["memory_summary"] = cs.get("memory_summary", s.get("memory_summary",""))
#                 s["awaiting"] = cs.get("awaiting", s.get("awaiting"))
#                 # hydrate messages from convos collection
#                 try:
#                     rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1).limit(200))
#                     if rows:
#                         msgs = []
#                         for r in rows:
#                             msgs.append({
#                                 "user": r.get("user_message"),
#                                 "assistant": r.get("bot_response"),
#                                 "agent": r.get("agent_used"),
#                                 "timestamp": r.get("timestamp")
#                             })
#                         s["messages"] = msgs
#                 except Exception:
#                     pass
#                 self.sessions[session_id] = s
#                 # persist to ensure db/memory alignment
#                 try:
#                     persist_session_state_raw(s.get("user_email", user_email) or user_email, session_id, s)
#                 except Exception:
#                     pass
#                 return True

#             # If not found, try to find the user by email and use their current_session (handles mismatch cases)
#             if user_email:
#                 u2 = users_col.find_one({"email": user_email})
#                 if u2 and u2.get("current_session") and u2["current_session"].get("session_id"):
#                     real_sid = u2["current_session"].get("session_id")
#                     # If the real_sid is the same as the provided, we've already tried; otherwise load real_sid
#                     if real_sid and real_sid not in self.sessions:
#                         return self.ensure_session_loaded(real_sid, user_email)

#             # Fallback: search conversations collection for provided session_id
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1).limit(200))
#             if rows:
#                 first = rows[0]
#                 s = self._new_session(first.get("user_email","") or user_email)
#                 msgs = []
#                 for r in rows:
#                     msgs.append({
#                         "user": r.get("user_message"),
#                         "assistant": r.get("bot_response"),
#                         "agent": r.get("agent_used"),
#                         "timestamp": r.get("timestamp")
#                     })
#                 s["messages"] = msgs
#                 self.sessions[session_id] = s
#                 try:
#                     persist_session_state_raw(s.get("user_email", user_email) or user_email, session_id, s)
#                 except Exception:
#                     pass
#                 return True

#             # Last resort: find recent conversation rows by user_email and load that session
#             if user_email:
#                 r = convos_col.find_one({"user_email": user_email}, sort=[("timestamp", -1)])
#                 if r and r.get("session_id"):
#                     return self.ensure_session_loaded(r.get("session_id"), user_email)

#         except Exception as e:
#             print("[ensure_session_loaded error]", e, file=sys.stderr)
#         return False

#     def get_or_create_session(self, user_email: str, session_id: Optional[str] = None) -> str:
#         if session_id:
#             # Try to rehydrate an existing session from DB first (robust)
#             try:
#                 loaded = self.ensure_session_loaded(session_id, user_email)
#                 if not loaded:
#                     # create an empty session but persist it (so restarts can rehydrate)
#                     self.sessions[session_id] = self._new_session(user_email)
#                     persist_session_state_raw(user_email, session_id, self.sessions[session_id])
#                 else:
#                     # already loaded by ensure_session_loaded
#                     pass
#                 # Merge any user-level current_session fields if present
#                 try:
#                     u = users_col.find_one({"email": user_email})
#                     if u and "current_session" in u and u["current_session"].get("session_id") == session_id:
#                         cs = u["current_session"]
#                         s = self.sessions[session_id]
#                         s["stage"] = cs.get("stage", s["stage"])
#                         s["selected_vehicle"] = cs.get("selected_vehicle", s["selected_vehicle"])
#                         s["order_id"] = cs.get("order_id", s.get("order_id"))
#                         s["collected"] = cs.get("collected", s.get("collected", {}))
#                 except Exception:
#                     pass
#             except Exception:
#                 # fallback create
#                 self.sessions[session_id] = self._new_session(user_email)
#                 persist_session_state_raw(user_email, session_id, self.sessions[session_id])
#             return session_id
#         sid = f"{user_email}_{int(time.time())}"
#         self.sessions[sid] = self._new_session(user_email)
#         persist_session_state_raw(user_email, sid, self.sessions[sid])
#         return sid

#     def add_message(self, session_id: str, user_message: str, bot_response: str, agent_used: str):
#         if session_id not in self.sessions:
#             self.sessions[session_id] = self._new_session("")
#         user_message = sanitize_text(user_message, max_len=4000)
#         bot_response = sanitize_text(bot_response, max_len=4000)
#         entry = {"user": user_message, "assistant": bot_response, "agent": agent_used, "timestamp": utcnow_iso()}
#         self.sessions[session_id]["messages"].append(entry)
#         try:
#             conv_doc = {
#                 "session_id": session_id,
#                 "user_email": self.sessions[session_id].get("user_email", ""),
#                 "user_message": user_message,
#                 "bot_response": bot_response,
#                 "agent_used": agent_used,
#                 "timestamp": utcnow_iso(),
#                 "turn_index": len(self.sessions[session_id]["messages"]) - 1
#             }
#             conv_doc_safe = _make_json_safe(conv_doc)
#             convos_col.insert_one(conv_doc_safe)
#         except Exception as e:
#             print("[convos_col insert error]", e, file=sys.stderr)
#             try:
#                 failed_writes_col.insert_one({"collection": "conversations", "error": str(e), "doc": _make_json_safe(conv_doc), "timestamp": utcnow_iso()})
#             except Exception:
#                 pass
#         if self.checkpointer:
#             try:
#                 config = {"configurable": {"thread_id": session_id}}
#                 state = {"messages": [HumanMessage(content=user_message), AIMessage(content=bot_response)]}
#                 try:
#                     self.checkpointer.put(config, state, {})
#                 except TypeError:
#                     self.checkpointer.put(config, state)
#             except Exception:
#                 pass
#         if self.store:
#             try:
#                 namespace = ("conversations", session_id)
#                 key = f"msg_{len(self.sessions[session_id]['messages'])}"
#                 try:
#                     self.store.put(namespace, key, entry)
#                 except TypeError:
#                     self.store.put(namespace, key, entry, {})
#             except Exception:
#                 pass
#         try:
#             persist_session_state(session_id)
#         except Exception as e:
#             print("[persist_session_state error]", e, file=sys.stderr)

#     def get_session_messages(self, session_id: str) -> List[Dict[str,Any]]:
#         if session_id in self.sessions and self.sessions[session_id]["messages"]:
#             return self.sessions[session_id]["messages"]
#         try:
#             rows = list(convos_col.find({"session_id": session_id}).sort("timestamp", 1))
#             if rows:
#                 out = []
#                 for r in rows:
#                     out.append({
#                         "user": r.get("user_message"),
#                         "assistant": r.get("bot_response"),
#                         "agent": r.get("agent_used"),
#                         "timestamp": r.get("timestamp")
#                     })
#                 if session_id not in self.sessions:
#                     self.sessions[session_id] = self._new_session(rows[0].get("user_email",""))
#                 self.sessions[session_id]["messages"] = out
#                 return out
#         except Exception as e:
#             print("[get_session_messages error]", e, file=sys.stderr)
#         if self.store is not None:
#             try:
#                 namespace = ("conversations", session_id)
#                 items = self.store.search(namespace)
#                 if items:
#                     return [it.value for it in items]
#             except Exception:
#                 pass
#         return []

#     def generate_summary(self, session_id: str) -> str:
#         msgs = self.get_session_messages(session_id)
#         if not msgs:
#             return "No messages to summarize."
#         convo_text = []
#         for m in msgs:
#             convo_text.append(f"User: {m.get('user')}")
#             convo_text.append(f"Assistant: {m.get('assistant')}")
#         prompt = ("Summarize the following conversation concisely. Include main topics, the selected vehicle (if chosen), and next steps.\n\n"
#                   "Conversation:\n" + "\n".join(convo_text) + "\n\nSummary:")
#         if llm:
#             try:
#                 resp = llm([{"role":"user","content":prompt}])
#                 summary = robust_extract_content(resp)
#             except Exception:
#                 summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         else:
#             summary = "Summary (fallback): " + " | ".join([m.get("user","")[:80] for m in msgs[:3]])
#         return sanitize_text(summary, max_len=1000)

#     def end_session_and_save(self, session_id: str):
#         if session_id not in self.sessions:
#             return "No session messages to summarize."
#         summary = self.generate_summary(session_id)
#         msgs = self.sessions[session_id]["messages"]
#         message_count = len(msgs)
#         start_time = self.sessions[session_id].get("start_time")
#         end_time = utcnow_iso()
#         user_email = self.sessions[session_id].get("user_email","")
#         try:
#             summaries_col.update_one(
#                 {"session_id": session_id},
#                 {"$set": {
#                     "session_id": session_id,
#                     "user_email": user_email,
#                     "summary": summary,
#                     "message_count": message_count,
#                     "start_time": start_time,
#                     "end_time": end_time,
#                     "created_at": utcnow_iso()
#                 }}, upsert=True
#             )
#             if user_email:
#                 users_col.update_one({"email": user_email},
#                                      {"$set": {"recent_summary": summary, "last_session_id": session_id}},
#                                      upsert=True)
#         except Exception as e:
#             print("[end_session_and_save error]", e, file=sys.stderr)
#         self.sessions[session_id]["stage"] = "finished"
#         try:
#             persist_session_state(session_id)
#         except Exception:
#             pass
#         return summary

# # ---------- Order helpers ----------
# def create_order_with_address(
#     session_id: str,
#     buyer_name: Optional[str] = None,
#     vehicle: Optional[Dict[str, Any]] = None,
#     sales_contact: Optional[Dict[str, Any]] = None,
#     buyer_address: Optional[str] = None,
#     buyer_phone: Optional[str] = None,
#     buyer_email: Optional[str] = None,
# ) -> Optional[str]:
#     session = memory_manager.sessions.get(session_id)
#     if not session:
#         raise ValueError("Invalid session_id or session not loaded. Make sure the session is rehydrated before placing an order.")

#     # Get buyer details from collected info
#     collected = session.get("collected", {})
#     if not buyer_name:
#         buyer_name = collected.get("name") or session.get("user_email")
#     if not buyer_address:
#         buyer_address = collected.get("address")
#     if not buyer_phone:
#         buyer_phone = collected.get("phone")
#     if not buyer_email:
#         buyer_email = collected.get("email") or session.get("user_email")

#     vehicle = normalize_vehicle(vehicle) or normalize_vehicle(session.get("selected_vehicle"))

#     if not vehicle:
#         raise ValueError("No vehicle selected for order")
#     if not buyer_address:
#         raise ValueError("Buyer address is required to place order")
#     if not isinstance(vehicle, dict):
#         raise ValueError("Selected vehicle data is invalid.")

#     order_doc = {
#         "session_id": session_id,
#         "user_email": session.get("user_email"),
#         "buyer_name": buyer_name,
#         "buyer_address": buyer_address,
#         "buyer_phone": buyer_phone,
#         "buyer_email": buyer_email,
#         "vehicle": {
#             "make": vehicle.get("make"),
#             "model": vehicle.get("model"),
#             "year": vehicle.get("year"),
#             "price": vehicle.get("price"),
#             "mileage": vehicle.get("mileage"),
#         },
#         "sales_contact": sales_contact or {
#             "name": "Jeni Flemin",
#             "position": "CEO",
#             "phone": "+94778540035",
#             "address": "Convent Garden, London, UK",
#         },
#         "timestamp": utcnow_iso(),
#         "order_date": date.today().isoformat(),
#         "conversation_summary": session.get("memory_summary", ""),
#     }

#     print(f"[create_order] Attempting to insert order: {json.dumps(order_doc, default=str, indent=2)}", file=sys.stderr)

#     try:
#         result = orders_col.insert_one(_make_json_safe(order_doc))
#         order_id = str(result.inserted_id)
#         session["order_id"] = order_id
#         session["stage"] = "ordered"
#         persist_session_state(session_id)
#         print(f"[create_order] Order created successfully with ID: {order_id}", file=sys.stderr)
#         return order_id
#     except Exception as e:
#         print(f"[create_order] Failed to insert order: {e}", file=sys.stderr)
#         try:
#             failed_writes_col.insert_one({
#                 "collection": "orders",
#                 "error": str(e),
#                 "doc": _make_json_safe(order_doc),
#                 "timestamp": utcnow_iso()
#             })
#         except Exception as e2:
#             print(f"[create_order] Failed to log to failed_writes: {e2}", file=sys.stderr)
#         raise

# memory_manager = ConversationMemoryManager()

# # ---------- Helpers ----------
# def persist_session_state(session_id: str):
#     s = memory_manager.sessions.get(session_id)
#     if not s:
#         return
#     email = s.get("user_email", "")
#     try:
#         users_col.update_one(
#             {"email": email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": s.get("stage"),
#                     "selected_vehicle": _make_json_safe(s.get("selected_vehicle")),
#                     "order_id": s.get("order_id"),
#                     "memory_summary": s.get("memory_summary", ""),
#                     "collected": _make_json_safe(s.get("collected", {})),
#                     "awaiting": s.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state]", e, file=sys.stderr)

# def persist_session_state_raw(user_email: str, session_id: str, session_obj: Dict[str,Any]):
#     try:
#         users_col.update_one(
#             {"email": user_email},
#             {"$set": {
#                 "current_session": {
#                     "session_id": session_id,
#                     "stage": session_obj.get("stage"),
#                     "selected_vehicle": _make_json_safe(session_obj.get("selected_vehicle")),
#                     "order_id": session_obj.get("order_id"),
#                     "memory_summary": session_obj.get("memory_summary", ""),
#                     "collected": _make_json_safe(session_obj.get("collected", {})),
#                     "awaiting": session_obj.get("awaiting"),
#                     "updated_at": utcnow_iso()
#                 },
#                 "last_session_id": session_id,
#                 "email": user_email
#             }},
#             upsert=True
#         )
#     except Exception as e:
#         print("[persist_session_state_raw]", e, file=sys.stderr)

# def fetch_user_profile_by_email(email: str) -> str:
#     if not email:
#         return "No email provided."
#     p = users_col.find_one({"email": email})
#     if not p:
#         return f"No profile found for {email}."
#     return f"Name: {p.get('name','')}\nEmail: {p.get('email','')}\nRecent summary: {p.get('recent_summary')}"

# # ... rest of your existing helper functions and code (tavily_search_raw, tool wrappers, etc.)
# # For brevity I keep these unchanged — in the actual file below keep the rest of your
# # previously provided code (tool definitions, agent creation, supervisor_invoke, etc.)

# # ---------- Note ----------
# # The canvas contains the full patched agent.py. Please replace your current file with
# # this version or merge the ensure_session_loaded method and the small call-site changes:
# #  - ConversationMemoryManager.ensure_session_loaded(...)
# #  - get_or_create_session uses ensure_session_loaded before creating a new empty session
# #  - call memory_manager.ensure_session_loaded(sid, user_email) inside handle_user_query
# #  - tool_place_order attempts rehydration before returning expired-session message

# # After updating, restart the service. Then test with your original session_id — it should
# # rehydrate the memory and allow order placement instead of returning the "expired" message.

# def fetch_cars_by_filters(filters: Dict[str,Any], limit: int = 10) -> List[Dict[str,Any]]:
#     q = {}
#     if "make" in filters:
#         q["make"] = {"$regex": re.compile(filters["make"], re.I)}
#     if "model" in filters:
#         q["model"] = {"$regex": re.compile(filters["model"], re.I)}
#     if "year_min" in filters or "year_max" in filters:
#         yq = {}
#         if "year_min" in filters: yq["$gte"] = int(filters["year_min"])
#         if "year_max" in filters: yq["$lte"] = int(filters["year_max"])
#         q["year"] = yq
#     if "price_min" in filters or "price_max" in filters:
#         pq = {}
#         if "price_min" in filters: pq["$gte"] = float(filters["price_min"])
#         if "price_max" in filters: pq["$lte"] = float(filters["price_max"])
#         q["price"] = pq
#     if "mileage_max" in filters:
#         q["mileage"] = {"$lte": int(filters["mileage_max"]) }
#     if "style" in filters:
#         q["style"] = {"$regex": re.compile(filters["style"], re.I)}
#     if "fuel_type" in filters:
#         q["fuel_type"] = {"$regex": re.compile(filters["fuel_type"], re.I)}
#     if "query" in filters:
#         q["$or"] = [
#             {"make": {"$regex": re.compile(filters["query"], re.I)}},
#             {"model": {"$regex": re.compile(filters["query"], re.I)}},
#             {"description": {"$regex": re.compile(filters["query"], re.I)}}
#         ]
#     cursor = cars_col.find(q).sort([("year",-1),("price",1)]).limit(limit)
#     return [c for c in cursor]

# def tavily_search_raw(q: str, max_results: int = 3) -> List[Dict[str,Any]]:
#     if not TAVILY_API_KEY:
#         return [{"error":"TAVILY_API_KEY not configured"}]
#     try:
#         from tavily import TavilyClient
#         client = TavilyClient(TAVILY_API_KEY)
#         response = client.search(query=q, time_range="month")
#         results = response.get("results", [])[:max_results]
#         return results
#     except Exception as e:
#         return [{"error": f"Tavily request failed: {e}"}]

# # ---------- Tooling helpers ----------
# CAR_JSON_MARKER = "===CAR_JSON==="
# WEB_JSON_MARKER = "===WEB_JSON==="

# def extract_and_store_json_markers_safe(text: str, session_id: str, memory_manager: ConversationMemoryManager):
#     """Extract JSON from markers and store in session - improved version"""
#     if not text:
#         return

#     def _parse_json_after_marker(after: str):
#         """Try multiple strategies to parse JSON"""
#         s = after.lstrip()

#         # Strategy 1: Try standard JSON decoder
#         decoder = json.JSONDecoder()
#         for start_char in ('{', '['):
#             idx = s.find(start_char)
#             if idx != -1:
#                 try:
#                     obj, _ = decoder.raw_decode(s[idx:])
#                     return obj
#                 except Exception:
#                     pass

#         # Strategy 2: Regex extraction
#         try:
#             m = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', s, re.DOTALL)
#             if m:
#                 return json.loads(m.group(1))
#         except Exception:
#             pass

#         # Strategy 3: Find balanced braces/brackets
#         for start_char, end_char in [('{', '}'), ('[', ']')]:
#             idx = s.find(start_char)
#             if idx != -1:
#                 depth = 0
#                 for i, c in enumerate(s[idx:], idx):
#                     if c == start_char:
#                         depth += 1
#                     elif c == end_char:
#                         depth -= 1
#                         if depth == 0:
#                             try:
#                                 return json.loads(s[idx:i+1])
#                             except Exception:
#                                 break
#         return None

#     # Extract CAR_JSON
#     if CAR_JSON_MARKER in text:
#         try:
#             after = text.split(CAR_JSON_MARKER, 1)[1]
#             parsed = _parse_json_after_marker(after)
#             if parsed is not None:
#                 s = memory_manager.sessions.setdefault(session_id, memory_manager._new_session(""))
#                 s['last_results'] = parsed
#                 persist_session_state(session_id)
#             else:
#                 print(f"[extract CAR_JSON] Failed to parse. Snippet: {after[:200]}", file=sys.stderr)
#         except Exception as e:
#             print(f"[extract CAR_JSON error] {e}", file=sys.stderr)

#     # Extract WEB_JSON
#     if WEB_JSON_MARKER in text:
#         try:
#             after = text.split(WEB_JSON_MARKER, 1)[1]
#             parsed = _parse_json_after_marker(after)
#             if parsed is not None:
#                 s = memory_manager.sessions.setdefault(session_id, memory_manager._new_session(""))
#                 s['last_web_results'] = parsed
#                 persist_session_state(session_id)
#             else:
#                 print(f"[extract WEB_JSON] Failed to parse. Snippet: {after[:200]}", file=sys.stderr)
#         except Exception as e:
#             print(f"[extract WEB_JSON error] {e}", file=sys.stderr)

# def robust_extract_content(response) -> str:
#     """Extract clean text content from various response formats"""
#     if response is None:
#         return ""
#     if isinstance(response, str):
#         return response

#     try:
#         # Handle LangChain agent responses (dict with 'messages' key)
#         if isinstance(response, dict) and "messages" in response:
#             messages = response["messages"]
#             if isinstance(messages, list) and len(messages) > 0:
#                 # Find the last AIMessage with actual content (not tool_calls)
#                 for msg in reversed(messages):
#                     # Skip ToolMessages
#                     is_tool_msg = (
#                         (hasattr(msg, '__class__') and msg.__class__.__name__ == 'ToolMessage') or
#                         (isinstance(msg, dict) and msg.get('type') == 'tool')
#                     )
#                     if is_tool_msg:
#                         continue

#                     # Check for AIMessage with content (and no tool_calls)
#                     has_tool_calls = False
#                     if hasattr(msg, 'tool_calls') and msg.tool_calls:
#                         has_tool_calls = True
#                     elif isinstance(msg, dict) and msg.get('tool_calls'):
#                         has_tool_calls = True

#                     # Extract content if it exists and no pending tool calls
#                     if not has_tool_calls:
#                         if hasattr(msg, 'content') and msg.content:
#                             return str(msg.content)
#                         if isinstance(msg, dict) and 'content' in msg and msg['content']:
#                             return str(msg['content'])

#         # Handle direct message objects with content attribute
#         if hasattr(response, "content"):
#             content = response.content
#             if isinstance(content, str) and content:
#                 return content
#             # Handle list of content blocks
#             if isinstance(content, list):
#                 text_parts = []
#                 for item in content:
#                     if isinstance(item, dict) and "text" in item:
#                         text_parts.append(item["text"])
#                     elif hasattr(item, "text"):
#                         text_parts.append(item.text)
#                     elif isinstance(item, str):
#                         text_parts.append(item)
#                 if text_parts:
#                     return "\n".join(text_parts)

#         # Handle OpenAI-style responses
#         if isinstance(response, dict):
#             if "choices" in response and len(response["choices"]) > 0:
#                 ch = response["choices"][0]
#                 if isinstance(ch, dict):
#                     if "message" in ch and "content" in ch["message"]:
#                         return ch["message"]["content"]
#                     if "text" in ch:
#                         return ch["text"]
#             # Handle direct content field
#             if "content" in response:
#                 content = response["content"]
#                 if isinstance(content, str) and content:
#                     return content

#         # Handle output attribute
#         if hasattr(response, "output"):
#             output = response.output
#             if isinstance(output, str):
#                 return output
#             # Recursively extract from output
#             return robust_extract_content(output)

#         # Last resort - convert to string but try to clean it
#         result = str(response)

#         # If it's a message dict dump, try to extract the last content
#         if ("'messages':" in result or '"messages":' in result) and len(result) > 200:
#             # Try to find the last content that's not empty
#             # Look for patterns like content='...' or content="..."
#             all_contents = re.findall(r"content=['\"]([^'\"]*(?:\\['\"][^'\"]*)*)['\"]", result)
#             if all_contents:
#                 # Return the last non-empty content
#                 for content in reversed(all_contents):
#                     if content and content.strip() and not content.startswith("HumanMessage"):
#                         return content.replace("\\'", "'").replace('\\"', '"')

#         return result
#     except Exception as e:
#         print(f"[robust_extract_content error] {e}", file=sys.stderr)
#         return str(response)

# def format_car_card(c: Dict[str, Any]) -> str:
#     """Format a single car as a readable line"""
#     if not c:
#         return "Unknown vehicle"
#     make = sanitize_text(str(c.get("make", "") or "Unknown"))
#     model = sanitize_text(str(c.get("model", "") or ""))
#     year = str(c.get("year", "") or "")
#     price = c.get("price")
#     if price is None or price == "":
#         price_str = "Price N/A"
#     else:
#         try:
#             if isinstance(price, (int, float)) and float(price).is_integer():
#                 price_str = f"${int(price):,}"
#             else:
#                 price_str = f"${float(price):,}"
#         except Exception:
#             price_str = str(price)
#     mileage = c.get("mileage")
#     mileage_str = f"{mileage} km" if mileage not in (None, "") else "Mileage N/A"
#     desc = sanitize_text(str(c.get("description", "") or ""))
#     if desc:
#         desc = desc.split(".")[0][:100]
#     title = " ".join(part for part in [make, model] if part).strip()
#     if year:
#         title = f"{title} ({year})"
#     return f"{title} — {price_str} — {mileage_str}" + (f" — {desc}" if desc else "")

# def build_results_message(cars: List[Dict[str, Any]]) -> str:
#     """Build human-readable car listing"""
#     if not cars:
#         return "No cars matched your filters."
#     total = len(cars)
#     top = cars[0] if total > 0 else None
#     lines = []
#     for i, c in enumerate(cars[:8], start=1):
#         lines.append(f"{i}. {format_car_card(c)}")
#     best_text = ""
#     if top:
#         make = top.get("make", "")
#         model = top.get("model", "")
#         year = top.get("year", "")
#         best_title = " ".join(part for part in [make, model] if part).strip()
#         if year:
#             best_title = f"{best_title} ({year})"
#         best_text = f"Top pick: {best_title}."
#     summary = f"I found {total} match{'es' if total != 1 else ''}. {best_text}\n"
#     summary += "Reply with the number to select a car, or say 'more filters' to narrow results."
#     return summary + "\n\n" + "\n".join(lines)

# # ---------- Tools ----------
# if LC_AVAILABLE:
#     @tool("get_user_profile", description="Fetch user profile (input: email). Returns profile text.")
#     def tool_get_user_profile(email: str) -> str:
#         return fetch_user_profile_by_email(email)

#     @tool("find_cars", description="Fetch cars in DB (input: JSON filters string). Returns formatted list with CAR_JSON marker.")
#     def tool_find_cars(filters_json: str) -> str:
#         try:
#             filters = json.loads(filters_json) if isinstance(filters_json, str) else {}
#         except Exception:
#             filters = {"query": filters_json}
#         cars = fetch_cars_by_filters(filters, limit=20)
#         out = []
#         for c in cars:
#             c2 = {k: v for k, v in c.items() if k != "_id"}
#             out.append(c2)

#         # Return CLEAN human text + JSON marker
#         if out:
#             human_text = build_results_message(out)
#             json_str = json.dumps(out, default=str)
#             return f"{human_text}\n\n{CAR_JSON_MARKER}{json_str}"
#         else:
#             return "I couldn't find any cars matching those filters. Try adjusting your criteria."

#     @tool("web_search", description="Search the web (input: query). Returns results with WEB_JSON marker.")
#     def tool_web_search(query: str) -> str:
#         results = tavily_search_raw(query, max_results=3)
#         human = "External search results:\n\n"
#         lines = []
#         for r in results:
#             if isinstance(r, dict) and r.get("error"):
#                 lines.append(r.get("error"))
#             else:
#                 title = r.get("title") or r.get("headline") or ""
#                 snippet = r.get("snippet") or r.get("summary") or ""
#                 url = r.get("url") or r.get("link") or ""
#                 lines.append(f"{title}\n{snippet}\n{url}")
#         human += "\n\n".join(lines) if lines else str(results)
#         return human + "\n\n" + WEB_JSON_MARKER + json.dumps(results, default=str)

#     @tool("place_order", description="Place order (input: JSON with session_id, buyer_address, etc.)")
#     def tool_place_order(payload: str) -> str:
#         try:
#             data = json.loads(payload)
#         except Exception:
#             return "Invalid payload. place_order expects JSON with session_id."
#         session_id = data.get("session_id")
#         if not session_id or session_id not in memory_manager.sessions:
#             return "Invalid or expired session_id. Please retry."
#         buyer_name = data.get("buyer_name")
#         vehicle = data.get("vehicle")
#         sales_contact = data.get("sales_contact")
#         buyer_address = data.get("buyer_address")
#         buyer_phone = data.get("buyer_phone")
#         buyer_email = data.get("buyer_email")

#         s = memory_manager.sessions.get(session_id)
#         if not vehicle and s:
#             vehicle = s.get("selected_vehicle")
#         try:
#             oid = create_order_with_address(
#                 session_id=session_id,
#                 buyer_name=buyer_name,
#                 vehicle=vehicle,
#                 sales_contact=sales_contact,
#                 buyer_address=buyer_address,
#                 buyer_phone=buyer_phone,
#                 buyer_email=buyer_email,
#             )
#             if oid:
#                 return f"✅ Order placed successfully! Order ID: {oid}"
#             else:
#                 return "Failed to place order (DB write returned None)."
#         except Exception as e:
#             print("[tool_place_order error]", e, file=sys.stderr)
#             try:
#                 failed_writes_col.insert_one({"collection": "orders", "error": str(e), "payload": _make_json_safe(data), "timestamp": utcnow_iso()})
#             except Exception:
#                 pass
#             return f"Error while placing order: {e}"

# # ---------- Create agents ----------
# personal_agent = None
# car_agent = None
# web_agent = None
# supervisor_agent = None

# if LC_AVAILABLE:
#     personal_prompt = (
#         "You are a Personal Agent. Fetch user profile when needed using get_user_profile tool.\n"
#         "Return clear, human-readable text."
#     )
#     personal_agent = create_agent(model=llm, name="PersonalAgent", system_prompt=personal_prompt, tools=[tool_get_user_profile])

#     car_prompt = (
#         "You are a Car Sales Agent. Search cars and place orders.\n"
#         "When returning results, use clean formatting followed by ===CAR_JSON=== marker with valid JSON.\n"
#         "Be helpful and conversational."
#     )
#     car_agent = create_agent(model=llm, name="CarSalesAgent", system_prompt=car_prompt, tools=[tool_find_cars, tool_place_order])

#     web_prompt = (
#         "You are a Web Agent. Search external sources using web_search tool when needed.\n"
#         "Return human-readable summaries."
#     )
#     web_agent = create_agent(model=llm, name="WebAgent", system_prompt=web_prompt, tools=[tool_web_search])

#     supervisor_system_prompt = (
#         "You are the Supervisor Agent for a car dealership. Be SIMPLE and DIRECT.\n\n"
#         "Rules:\n"
#         "1) Use car_wrapper to search cars and place orders\n"
#         "2) Use personal_wrapper for user profile info\n"
#         "3) Use web_wrapper only for external research if needed\n\n"
#         "Order Placement:\n"
#         "- When user says 'confirm', 'place order', 'buy', 'purchase' AND a vehicle is selected AND you have an address -> IMMEDIATELY place the order\n"
#         "- Don't ask for more details if you have: vehicle + address\n"
#         "- Don't complicate things - just place the order\n"
#         "- If missing address, ask ONCE for address only\n\n"
#         "Keep responses short and conversational. Don't over-explain."
#     )

#     @tool("personal_wrapper", description="Invoke Personal Agent")
#     def tool_personal_wrapper(payload: str) -> str:
#         if personal_agent:
#             try:
#                 resp = personal_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = personal_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Personal Agent error: {e}"
#         return "Personal Agent not available."

#     @tool("car_wrapper", description="Invoke Car Agent")
#     def tool_car_wrapper(payload: str) -> str:
#         if car_agent:
#             try:
#                 resp = car_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = car_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Car Agent error: {e}"
#         return "Car Agent not available."

#     @tool("web_wrapper", description="Invoke Web Agent")
#     def tool_web_wrapper(payload: str) -> str:
#         if web_agent:
#             try:
#                 resp = web_agent.invoke({"messages":[{"role":"user","content":payload}]})
#                 # Run to completion if needed
#                 max_iter = 5
#                 for _ in range(max_iter):
#                     if isinstance(resp, dict) and "messages" in resp:
#                         last_msg = resp["messages"][-1] if resp["messages"] else None
#                         if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                             resp = web_agent.invoke(resp)
#                         else:
#                             break
#                     else:
#                         break
#                 return robust_extract_content(resp)
#             except Exception as e:
#                 return f"Web Agent error: {e}"
#         return "Web Agent not available."

#     supervisor_agent = create_agent(model=llm, name="SupervisorAgent", system_prompt=supervisor_system_prompt, tools=[tool_personal_wrapper, tool_car_wrapper, tool_web_wrapper])

# # ---------- Helper functions ----------
# def is_order_confirmation(user_text: str) -> bool:
#     if not user_text:
#         return False
#     t = user_text.lower().strip()
#     keywords = ["confirm","place order","buy","purchase","i want this","proceed","yes i want","go ahead","yes"]
#     return any(k in t for k in keywords)

# def contains_address_info(user_text: str) -> bool:
#     """Check if user text contains address/contact information"""
#     if not user_text:
#         return False
#     t = user_text.lower()
#     address_indicators = ["address", "street", "road", "avenue", "city", "phone", "email", "name:"]
#     return any(indicator in t for indicator in address_indicators)

# def handle_car_selection(session_id: str, user_text: str) -> Optional[str]:
#     """Handle numeric car selection"""
#     s = memory_manager.sessions.get(session_id)
#     if not s:
#         return None
#     if not user_text or not user_text.strip().isdigit():
#         return None
#     if not s.get("last_results"):
#         return None
#     try:
#         idx = int(user_text.strip()) - 1
#         if idx < 0 or idx >= len(s["last_results"]):
#             return f"Selection {user_text.strip()} is out of range. Please choose between 1 and {len(s['last_results'])}."
#         sel = s["last_results"][idx]
#         try:
#             sel_copy = {k: (str(v) if k == "_id" else v) for k, v in sel.items()}
#         except Exception:
#             sel_copy = _make_json_safe(sel)
#         s["selected_vehicle"] = sel_copy
#         s["stage"] = "vehicle_selected"
#         s["awaiting"] = "address"  # Set awaiting status
#         persist_session_state(session_id)

#         response = (
#             f"✓ Great choice! You've selected:\n\n"
#             f"🚗 {sel_copy.get('make')} {sel_copy.get('model')} ({sel_copy.get('year')})\n"
#             f"💰 Price: ${sel_copy.get('price'):,}\n"
#             f"📏 Mileage: {sel_copy.get('mileage')} km\n\n"
#             f"To complete your order, please provide:\n"
#             f"• Your full name\n"
#             f"• Delivery address\n"
#             f"• Phone number\n"
#             f"• Email address\n\n"
#             f"You can provide them all at once or one at a time."
#         )
#         return response
#     except Exception as e:
#         print("[handle_car_selection]", e, file=sys.stderr)
#         return None

# # ---------- Main supervisor invoke ----------
# def supervisor_invoke(session_id: str, user_email: str, user_query: str) -> Tuple[str, str]:
#     """Main orchestrator with memory optimization and clean output"""
#     session = memory_manager.sessions.get(session_id, memory_manager._new_session(user_email))

#     # Extract contact info if present
#     contact_info = extract_contact_info(user_query)
#     if contact_info:
#         print(f"[supervisor_invoke] Extracted contact info: {contact_info}", file=sys.stderr)
#         collected = session.get("collected", {})
#         collected.update(contact_info)
#         session["collected"] = collected
#         persist_session_state(session_id)

#     # Check if we're awaiting address and user provided it
#     awaiting = session.get("awaiting")
#     selected_vehicle = session.get("selected_vehicle")
#     already_ordered = bool(session.get("order_id"))

#     # SIMPLIFIED ORDER PLACEMENT: Check if user wants to place order
#     if selected_vehicle and not already_ordered and is_order_confirmation(user_query):
#         print(f"[supervisor_invoke] Order confirmation detected", file=sys.stderr)
#         collected = session.get("collected", {})
#         buyer_address = collected.get("address")

#         # If we have address, place order immediately
#         if buyer_address:
#             print(f"[supervisor_invoke] Have address, placing order now", file=sys.stderr)
#             try:
#                 oid = create_order_with_address(
#                     session_id=session_id,
#                     buyer_name=collected.get("name") or user_email,
#                     vehicle=selected_vehicle,
#                     sales_contact={
#                         "name": "Jeni Flemin",
#                         "position": "CEO",
#                         "phone": "+94778540035",
#                         "address": "Convent Garden, London, UK"
#                     },
#                     buyer_address=buyer_address,
#                     buyer_phone=collected.get("phone"),
#                     buyer_email=collected.get("email") or user_email
#                 )

#                 if oid:
#                     session["awaiting"] = None
#                     persist_session_state(session_id)

#                     success_msg = (
#                         f"✅ Order placed successfully!\n\n"
#                         f"Order ID: {oid}\n"
#                         f"Vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})\n"
#                         f"Price: ${selected_vehicle.get('price'):,}\n"
#                         f"Delivery to: {buyer_address}\n\n"
#                         f"Our team will contact you within 24 hours."
#                     )
#                     memory_manager.add_message(session_id, user_query, success_msg, agent_used="OrderHandler")
#                     return success_msg, session_id

#             except Exception as e:
#                 error_msg = f"Sorry, there was an error placing your order: {str(e)}. Please try again."
#                 print(f"[supervisor_invoke] Order placement error: {e}", file=sys.stderr)
#                 memory_manager.add_message(session_id, user_query, error_msg, agent_used="OrderHandler")
#                 return error_msg, session_id
#         else:
#             # Ask for address ONCE
#             session["awaiting"] = "address"
#             persist_session_state(session_id)
#             ask_text = (
#                 f"To complete your order for the {selected_vehicle.get('make')} {selected_vehicle.get('model')}, "
#                 f"I just need your delivery address.\n\n"
#                 f"Please provide your full address."
#             )
#             memory_manager.add_message(session_id, user_query, ask_text, agent_used="OrderHandler")
#             return ask_text, session_id

#     # If awaiting address and user provides address info, place order
#     if awaiting == "address" and contains_address_info(user_query) and selected_vehicle and not already_ordered:
#         print(f"[supervisor_invoke] Address provided, attempting order placement", file=sys.stderr)
#         collected = session.get("collected", {})
#         buyer_address = collected.get("address")

#         if buyer_address:
#             try:
#                 oid = create_order_with_address(
#                     session_id=session_id,
#                     buyer_name=collected.get("name") or user_email,
#                     vehicle=selected_vehicle,
#                     sales_contact={
#                         "name": "Jeni Flemin",
#                         "position": "CEO",
#                         "phone": "+94778540035",
#                         "address": "Convent Garden, London, UK"
#                     },
#                     buyer_address=buyer_address,
#                     buyer_phone=collected.get("phone"),
#                     buyer_email=collected.get("email") or user_email
#                 )

#                 if oid:
#                     session["awaiting"] = None
#                     persist_session_state(session_id)

#                     success_msg = (
#                         f"✅ Order placed successfully!\n\n"
#                         f"Order ID: {oid}\n"
#                         f"Vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})\n"
#                         f"Price: ${selected_vehicle.get('price'):,}\n"
#                         f"Delivery to: {buyer_address}\n\n"
#                         f"Our team will contact you within 24 hours."
#                     )
#                     memory_manager.add_message(session_id, user_query, success_msg, agent_used="OrderHandler")
#                     return success_msg, session_id

#             except Exception as e:
#                 error_msg = f"Sorry, there was an error placing your order: {str(e)}. Please try again."
#                 print(f"[supervisor_invoke] Order placement error: {e}", file=sys.stderr)
#                 memory_manager.add_message(session_id, user_query, error_msg, agent_used="OrderHandler")
#                 return error_msg, session_id

#     # Handle numeric selection first
#     sel_reply = handle_car_selection(session_id, user_query)
#     if sel_reply is not None:
#         memory_manager.add_message(session_id, user_query, sel_reply, agent_used="SelectionHandler")
#         return sel_reply, session_id

#     # Build context with memory optimization
#     conversation_context = memory_manager.get_context_for_llm(session_id) if hasattr(memory_manager, 'get_context_for_llm') else None

#     # Add session state context
#     state_context = ""
#     if selected_vehicle and not already_ordered:
#         state_context = f"\n[Selected vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')}) - ${selected_vehicle.get('price'):,}]"
#         collected = session.get("collected", {})
#         if collected.get("address"):
#             state_context += f"\n[Have address: {collected.get('address')}]"
#             state_context += "\n[READY TO PLACE ORDER - User just needs to say 'confirm' or 'place order']"
#         else:
#             state_context += "\n[Need address to place order]"

#     full_prompt = f"\nPrevious conversation:\n{conversation_context}\n{state_context}\n\nUser: {user_query}\n" if conversation_context else user_query

#     # Get response from supervisor - run agent to completion
#     if LC_AVAILABLE and supervisor_agent:
#         try:
#             # Run agent with proper message format
#             messages = [{"role": "user", "content": full_prompt}]
#             resp = supervisor_agent.invoke({"messages": messages})

#             # Keep invoking until we get a final response (no tool_calls in last message)
#             max_iterations = 10
#             iteration = 0

#             while iteration < max_iterations:
#                 iteration += 1

#                 # Check if response has messages
#                 if isinstance(resp, dict) and "messages" in resp:
#                     messages_list = resp["messages"]
#                     if not messages_list:
#                         break

#                     last_msg = messages_list[-1]

#                     # Check if last message has tool_calls (needs to continue)
#                     has_tool_calls = False
#                     if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
#                         has_tool_calls = True
#                     elif isinstance(last_msg, dict) and last_msg.get('tool_calls'):
#                         has_tool_calls = True

#                     # If the last message is a ToolMessage, we need another iteration for AI to respond
#                     is_tool_message = (
#                         (hasattr(last_msg, '__class__') and last_msg.__class__.__name__ == 'ToolMessage') or
#                         (isinstance(last_msg, dict) and last_msg.get('type') == 'tool')
#                     )

#                     # Continue if we have tool calls OR if last message is a tool result
#                     if has_tool_calls or is_tool_message:
#                         try:
#                             resp = supervisor_agent.invoke(resp)
#                         except Exception as e:
#                             print(f"[agent continuation error] {e}", file=sys.stderr)
#                             break
#                     else:
#                         # Last message is AIMessage with content, we're done
#                         break
#                 else:
#                     # Not in expected format, break
#                     break

#             out_raw = robust_extract_content(resp)

#         except Exception as e:
#             print(f"[supervisor error] {e}", file=sys.stderr)
#             import traceback
#             traceback.print_exc(file=sys.stderr)
#             out_raw = f"Sorry, I encountered an error: {e}"
#     else:
#         try:
#             resp = llm([{"role": "user", "content": full_prompt}])
#             out_raw = robust_extract_content(resp)
#         except Exception as e:
#             out_raw = f"Fallback response: {user_query} (error: {e})"

#     # Extract JSON markers into session
#     try:
#         extract_and_store_json_markers_safe(str(out_raw), session_id, memory_manager)
#     except Exception as e:
#         print("[extract_and_store_json_markers error]", e, file=sys.stderr)

#     # Clean the output - remove JSON markers from user-facing text
#     cleaned_output = out_raw
#     if CAR_JSON_MARKER in cleaned_output:
#         cleaned_output = cleaned_output.split(CAR_JSON_MARKER)[0]
#     if WEB_JSON_MARKER in cleaned_output:
#         cleaned_output = cleaned_output.split(WEB_JSON_MARKER)[0]
#     cleaned_output = cleaned_output.strip()

#     # Save to conversation history
#     memory_manager.add_message(session_id, user_query, cleaned_output, agent_used="Supervisor")

#     return cleaned_output, session_id

# # ---------- Top-level API ----------
# def handle_user_query(session_id: Optional[str], user_email: str, user_query: str) -> Dict[str,Any]:
#     """Main entry point for handling user queries"""
#     sid = memory_manager.get_or_create_session(user_email, session_id)
#     memory_manager.sessions[sid]["user_email"] = user_email
#     resp, _ = supervisor_invoke(sid, user_email, user_query)
#     return {"response": resp, "session_id": sid}

# def end_session(session_id: str) -> str:
#     """End session and generate summary"""
#     return memory_manager.end_session_and_save(session_id)

# __all__ = [
#     'handle_user_query',
#     'end_session',
#     'memory_manager',
#     'create_order_with_address',
#     'persist_session_state'
# ]


import os
import re
import json
import time
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, date
from decimal import Decimal
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

if not MONGO_URI:
    raise RuntimeError("MONGODB_CONNECTION_STRING not set in .env")

# ---------- MongoDB ----------
from pymongo import MongoClient

mongo = MongoClient(MONGO_URI)
db = mongo.get_database()
cars_col = db.get_collection("cars")
users_col = db.get_collection("users")
convos_col = db.get_collection("conversations")
summaries_col = db.get_collection("conversation_summaries")
orders_col = db.get_collection("orders")
failed_writes_col = db.get_collection("failed_writes")

# ---------- LangGraph Memory ----------
try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langchain_core.messages import HumanMessage, AIMessage

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False
    InMemorySaver = InMemoryStore = None

    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class AIMessage:
        def __init__(self, content: str):
            self.content = content


# ---------- LangChain / OpenAI detection ----------
llm = None
LC_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langchain.tools import tool

    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
    LC_AVAILABLE = True
except Exception:
    import openai

    openai.api_key = OPENAI_API_KEY

    class SimpleOpenAIWrapper:
        def __init__(self, model=LLM_MODEL_NAME, temperature=0):
            self.model = model
            self.temperature = temperature

        def __call__(self, messages: List[Dict[str, str]]):
            return openai.ChatCompletion.create(
                model=self.model, messages=messages, temperature=self.temperature
            )

    llm = SimpleOpenAIWrapper()
    LC_AVAILABLE = False


# ---------- Utilities ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(s: str, max_len: int = 4000) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _make_json_safe(obj: Any) -> Any:
    """
    Convert object to MongoDB-safe format while preserving types.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    try:
        return str(obj)
    except Exception:
        return None


def normalize_vehicle(vehicle):
    """Ensure vehicle is a dict."""
    if not vehicle:
        return None
    if isinstance(vehicle, dict):
        return vehicle
    return None


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract contact information from user input"""
    info = {}
    name_match = re.search(r"name\s*:?\s*([^,\n]+)", text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(1).strip()
    phone_match = re.search(r"phone\s*:?\s*([\+\d\s\-\(\)]+)", text, re.IGNORECASE)
    if phone_match:
        info["phone"] = phone_match.group(1).strip()
    email_match = re.search(r"email\s*:?\s*([^\s,]+@[^\s,]+)", text, re.IGNORECASE)
    if email_match:
        info["email"] = email_match.group(1).strip()
    address_match = re.search(
        r"(?:delivery\s+)?address\s*:?\s*([^,]+(?:,[^,]+)*)", text, re.IGNORECASE
    )
    if address_match:
        info["address"] = address_match.group(1).strip()
    return info


# [Continue with MemoryOptimizerMixin and ConversationMemoryManager - unchanged from your original]
class MemoryOptimizerMixin:
    MAX_PROMPT_TOKENS = 3000
    RECENT_TURNS_KEEP = 8
    SUMMARIZE_EVERY = 12
    SUMMARY_MAX_TOKENS = 800

    def compress_history_if_needed(self, session_id: str):
        s = self.sessions.get(session_id)
        if not s:
            return
        msgs = s.get("messages", [])
        if len(msgs) <= (self.RECENT_TURNS_KEEP + 2):
            return
        last_summary_at = s.get("_last_summary_index", 0)
        if len(msgs) - last_summary_at < self.SUMMARIZE_EVERY:
            return
        older = msgs[: max(0, len(msgs) - self.RECENT_TURNS_KEEP)]
        if not older:
            return
        older_text = []
        for m in older:
            u = m.get("user") or ""
            a = m.get("assistant") or ""
            if u:
                older_text.append(f"User: {sanitize_text(u, 2000)}")
            if a:
                older_text.append(f"Assistant: {sanitize_text(a, 2000)}")
        to_summarize = "\n".join(older_text)
        if estimate_tokens(to_summarize) < (self.SUMMARY_MAX_TOKENS // 2):
            summary = to_summarize
        else:
            prompt = (
                "Summarize the following conversation history into concise bullet points. "
                "Keep facts, decisions, selected vehicle details, outstanding questions and next steps. "
                "Limit to ~200-400 words.\n\nHistory:\n" + to_summarize + "\n\nSummary:"
            )
            try:
                resp = llm([{"role": "user", "content": prompt}])
                summary = robust_extract_content(resp)
                if not summary:
                    summary = "(summary generation failed)"
            except Exception as e:
                summary = f"(summary generation failed: {e})"
        prev = s.get("memory_summary", "") or ""
        new_summary = (prev + "\n---\n" + summary) if prev else summary
        recent = msgs[-self.RECENT_TURNS_KEEP :]
        placeholder = {
            "user": "[older history summarized]",
            "assistant": new_summary,
            "agent": "system_summary",
            "timestamp": utcnow_iso(),
        }
        s["messages"] = [placeholder] + recent
        s["_last_summary_index"] = len(s["messages"])
        s["memory_summary"] = new_summary
        try:
            persist_session_state(session_id)
        except Exception as e:
            print("[compress_history persist error]", e, file=sys.stderr)

    def get_context_for_llm(self, session_id: str, max_messages: int = None) -> str:
        s = self.sessions.get(session_id)
        if not s:
            return ""
        try:
            self.compress_history_if_needed(session_id)
        except Exception:
            pass
        memory_summary = s.get("memory_summary", "") or ""
        recent = s.get("messages", [])[-self.RECENT_TURNS_KEEP :]
        lines = []
        tokens_used = 0
        if memory_summary:
            ts = f"Memory Summary:\n{memory_summary}\n"
            t_count = estimate_tokens(ts)
            lines.append(ts)
            tokens_used += t_count
        for m in recent:
            u = m.get("user") or ""
            a = m.get("assistant") or ""
            agent = m.get("agent") or ""
            if u:
                line = f"User: {u}\n"
                t = estimate_tokens(line)
                if tokens_used + t > self.MAX_PROMPT_TOKENS:
                    break
                lines.append(line)
                tokens_used += t
            if a:
                line = f"Assistant ({agent}): {a}\n"
                t = estimate_tokens(line)
                if tokens_used + t > self.MAX_PROMPT_TOKENS:
                    break
                lines.append(line)
                tokens_used += t
        return "\n".join(lines)


class ConversationMemoryManager(MemoryOptimizerMixin):
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
        if not session_id:
            return False
        if session_id in self.sessions:
            return True
        try:
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
                    persist_session_state_raw(
                        s.get("user_email", user_email) or user_email, session_id, s
                    )
                except Exception:
                    pass
                return True
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
                    persist_session_state_raw(
                        s.get("user_email", user_email) or user_email, session_id, s
                    )
                except Exception:
                    pass
                return True
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
        if session_id:
            try:
                loaded = self.ensure_session_loaded(session_id, user_email)
                if not loaded:
                    self.sessions[session_id] = self._new_session(user_email)
                    persist_session_state_raw(
                        user_email, session_id, self.sessions[session_id]
                    )
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
                persist_session_state_raw(
                    user_email, session_id, self.sessions[session_id]
                )
            return session_id
        sid = f"{user_email}_{int(time.time())}"
        self.sessions[sid] = self._new_session(user_email)
        persist_session_state_raw(user_email, sid, self.sessions[sid])
        return sid

    def add_message(
        self, session_id: str, user_message: str, bot_response: str, agent_used: str
    ):
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
        try:
            persist_session_state(session_id)
        except Exception as e:
            print("[persist_session_state error]", e, file=sys.stderr)

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id in self.sessions and self.sessions[session_id]["messages"]:
            return self.sessions[session_id]["messages"]
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
        msgs = self.get_session_messages(session_id)
        if not msgs:
            return "No messages to summarize."
        convo_text = []
        for m in msgs:
            convo_text.append(f"User: {m.get('user')}")
            convo_text.append(f"Assistant: {m.get('assistant')}")
        prompt = (
            "Summarize the following conversation concisely. Include main topics, the selected vehicle (if chosen), and next steps.\n\n"
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
            persist_session_state(session_id)
        except Exception:
            pass
        return summary


# ---------- Order helpers ----------
def create_order_with_address(
    session_id: str,
    buyer_name: Optional[str] = None,
    vehicle: Optional[Dict[str, Any]] = None,
    sales_contact: Optional[Dict[str, Any]] = None,
    buyer_address: Optional[str] = None,
    buyer_phone: Optional[str] = None,
    buyer_email: Optional[str] = None,
) -> Optional[str]:
    """Create an order with improved error handling"""
    session = memory_manager.sessions.get(session_id)
    if not session:
        print(f"[create_order] ERROR: Session {session_id} not found", file=sys.stderr)
        try:
            memory_manager.ensure_session_loaded(session_id, buyer_email or "")
            session = memory_manager.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        except Exception as e:
            print(f"[create_order] ERROR: Failed to load session: {e}", file=sys.stderr)
            raise ValueError(f"Invalid session_id: {e}")

    collected = session.get("collected", {})
    if not buyer_name:
        buyer_name = collected.get("name") or session.get("user_email", "Unknown")
    if not buyer_address:
        buyer_address = collected.get("address")
    if not buyer_phone:
        buyer_phone = collected.get("phone")
    if not buyer_email:
        buyer_email = collected.get("email") or session.get("user_email")

    if not buyer_address:
        raise ValueError("Buyer address is required")

    if not vehicle:
        vehicle = session.get("selected_vehicle")

    if not vehicle:
        raise ValueError("No vehicle selected")

    if not isinstance(vehicle, dict):
        raise ValueError("Invalid vehicle data")

    # Clean vehicle data
    vehicle_clean = {}
    for key in [
        "make",
        "model",
        "year",
        "price",
        "mileage",
        "style",
        "fuel_type",
        "description",
    ]:
        if key in vehicle:
            val = vehicle[key]
            if isinstance(val, ObjectId):
                vehicle_clean[key] = str(val)
            elif isinstance(val, (int, float)):
                vehicle_clean[key] = val
            else:
                vehicle_clean[key] = str(val) if val is not None else None

    # Ensure numeric types
    if "price" in vehicle_clean and vehicle_clean["price"]:
        try:
            vehicle_clean["price"] = float(vehicle_clean["price"])
        except (ValueError, TypeError):
            print(f"[create_order] WARNING: Could not convert price", file=sys.stderr)

    if "year" in vehicle_clean and vehicle_clean["year"]:
        try:
            vehicle_clean["year"] = int(vehicle_clean["year"])
        except (ValueError, TypeError):
            print(f"[create_order] WARNING: Could not convert year", file=sys.stderr)

    if "mileage" in vehicle_clean and vehicle_clean["mileage"]:
        try:
            vehicle_clean["mileage"] = int(vehicle_clean["mileage"])
        except (ValueError, TypeError):
            print(f"[create_order] WARNING: Could not convert mileage", file=sys.stderr)

    if not sales_contact:
        sales_contact = {
            "name": "Jeni Flemin",
            "position": "CEO",
            "phone": "+94778540035",
            "address": "Convent Garden, London, UK",
        }

    order_doc = {
        "session_id": session_id,
        "user_email": session.get("user_email", buyer_email),
        "buyer_name": buyer_name,
        "buyer_address": buyer_address,
        "buyer_phone": buyer_phone,
        "buyer_email": buyer_email,
        "vehicle": vehicle_clean,
        "sales_contact": sales_contact,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_date": date.today().isoformat(),
        "conversation_summary": session.get("memory_summary", ""),
        "status": "pending",
    }

    print(f"[create_order] Inserting order for {buyer_name}", file=sys.stderr)
    print(
        f"[create_order] Vehicle: {vehicle_clean.get('make')} {vehicle_clean.get('model')}",
        file=sys.stderr,
    )

    try:
        result = orders_col.insert_one(order_doc)

        if result and result.inserted_id:
            order_id = str(result.inserted_id)
            session["order_id"] = order_id
            session["stage"] = "ordered"

            try:
                persist_session_state(session_id)
            except Exception as persist_error:
                print(
                    f"[create_order] WARNING: Persist failed: {persist_error}",
                    file=sys.stderr,
                )

            print(f"[create_order] ✅ Order {order_id} created", file=sys.stderr)

            try:
                verify = orders_col.find_one({"_id": result.inserted_id})
                if verify:
                    print(f"[create_order] ✅ Order verified", file=sys.stderr)
                else:
                    print(
                        f"[create_order] ⚠️ Order not found after insert",
                        file=sys.stderr,
                    )
            except Exception:
                pass

            return order_id
        else:
            raise Exception("Insert returned no ID")

    except Exception as e:
        print(f"[create_order] ❌ ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)

        try:
            failed_writes_col.insert_one(
                {
                    "collection": "orders",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "doc": order_doc,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                }
            )
        except Exception as log_error:
            print(f"[create_order] Failed to log error: {log_error}", file=sys.stderr)

        raise


memory_manager = ConversationMemoryManager()


# ---------- Helpers ----------
def persist_session_state(session_id: str):
    s = memory_manager.sessions.get(session_id)
    if not s:
        return
    email = s.get("user_email", "")
    try:
        users_col.update_one(
            {"email": email},
            {
                "$set": {
                    "current_session": {
                        "session_id": session_id,
                        "stage": s.get("stage"),
                        "selected_vehicle": _make_json_safe(s.get("selected_vehicle")),
                        "order_id": s.get("order_id"),
                        "memory_summary": s.get("memory_summary", ""),
                        "collected": _make_json_safe(s.get("collected", {})),
                        "awaiting": s.get("awaiting"),
                        "updated_at": utcnow_iso(),
                    },
                    "last_session_id": session_id,
                    "email": email,
                }
            },
            upsert=True,
        )
    except Exception as e:
        print("[persist_session_state]", e, file=sys.stderr)


def persist_session_state_raw(
    user_email: str, session_id: str, session_obj: Dict[str, Any]
):
    try:
        users_col.update_one(
            {"email": user_email},
            {
                "$set": {
                    "current_session": {
                        "session_id": session_id,
                        "stage": session_obj.get("stage"),
                        "selected_vehicle": _make_json_safe(
                            session_obj.get("selected_vehicle")
                        ),
                        "order_id": session_obj.get("order_id"),
                        "memory_summary": session_obj.get("memory_summary", ""),
                        "collected": _make_json_safe(session_obj.get("collected", {})),
                        "awaiting": session_obj.get("awaiting"),
                        "updated_at": utcnow_iso(),
                    },
                    "last_session_id": session_id,
                    "email": user_email,
                }
            },
            upsert=True,
        )
    except Exception as e:
        print("[persist_session_state_raw]", e, file=sys.stderr)


def fetch_user_profile_by_email(email: str) -> str:
    if not email:
        return "No email provided."
    p = users_col.find_one({"email": email})
    if not p:
        return f"No profile found for {email}."
    return f"Name: {p.get('name','')}\nEmail: {p.get('email','')}\nRecent summary: {p.get('recent_summary')}"


def fetch_cars_by_filters(
    filters: Dict[str, Any], limit: int = 10
) -> List[Dict[str, Any]]:
    q = {}
    if "make" in filters:
        q["make"] = {"$regex": re.compile(filters["make"], re.I)}
    if "model" in filters:
        q["model"] = {"$regex": re.compile(filters["model"], re.I)}
    if "year_min" in filters or "year_max" in filters:
        yq = {}
        if "year_min" in filters:
            yq["$gte"] = int(filters["year_min"])
        if "year_max" in filters:
            yq["$lte"] = int(filters["year_max"])
        q["year"] = yq
    if "price_min" in filters or "price_max" in filters:
        pq = {}
        if "price_min" in filters:
            pq["$gte"] = float(filters["price_min"])
        if "price_max" in filters:
            pq["$lte"] = float(filters["price_max"])
        q["price"] = pq
    if "mileage_max" in filters:
        q["mileage"] = {"$lte": int(filters["mileage_max"])}
    if "style" in filters:
        q["style"] = {"$regex": re.compile(filters["style"], re.I)}
    if "fuel_type" in filters:
        q["fuel_type"] = {"$regex": re.compile(filters["fuel_type"], re.I)}
    if "query" in filters:
        q["$or"] = [
            {"make": {"$regex": re.compile(filters["query"], re.I)}},
            {"model": {"$regex": re.compile(filters["query"], re.I)}},
            {"description": {"$regex": re.compile(filters["query"], re.I)}},
        ]
    cursor = cars_col.find(q).sort([("year", -1), ("price", 1)]).limit(limit)
    return [c for c in cursor]


def tavily_search_raw(q: str, max_results: int = 3) -> List[Dict[str, Any]]:
    if not TAVILY_API_KEY:
        return [{"error": "TAVILY_API_KEY not configured"}]
    try:
        from tavily import TavilyClient

        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(query=q, time_range="month")
        results = response.get("results", [])[:max_results]
        return results
    except Exception as e:
        return [{"error": f"Tavily request failed: {e}"}]


CAR_JSON_MARKER = "===CAR_JSON==="
WEB_JSON_MARKER = "===WEB_JSON==="


def extract_and_store_json_markers_safe(
    text: str, session_id: str, memory_manager: ConversationMemoryManager
):
    if not text:
        return

    def _parse_json_after_marker(after: str):
        s = after.lstrip()
        decoder = json.JSONDecoder()
        for start_char in ("{", "["):
            idx = s.find(start_char)
            if idx != -1:
                try:
                    obj, _ = decoder.raw_decode(s[idx:])
                    return obj
                except Exception:
                    pass
        try:
            m = re.search(r"(\{[^{}]*\}|\[[^\[\]]*\])", s, re.DOTALL)
            if m:
                return json.loads(m.group(1))
        except Exception:
            pass
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            idx = s.find(start_char)
            if idx != -1:
                depth = 0
                for i, c in enumerate(s[idx:], idx):
                    if c == start_char:
                        depth += 1
                    elif c == end_char:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(s[idx : i + 1])
                            except Exception:
                                break
        return None

    if CAR_JSON_MARKER in text:
        try:
            after = text.split(CAR_JSON_MARKER, 1)[1]
            parsed = _parse_json_after_marker(after)
            if parsed is not None:
                s = memory_manager.sessions.setdefault(
                    session_id, memory_manager._new_session("")
                )
                s["last_results"] = parsed
                persist_session_state(session_id)
        except Exception as e:
            print(f"[extract CAR_JSON error] {e}", file=sys.stderr)

    if WEB_JSON_MARKER in text:
        try:
            after = text.split(WEB_JSON_MARKER, 1)[1]
            parsed = _parse_json_after_marker(after)
            if parsed is not None:
                s = memory_manager.sessions.setdefault(
                    session_id, memory_manager._new_session("")
                )
                s["last_web_results"] = parsed
                persist_session_state(session_id)
        except Exception as e:
            print(f"[extract WEB_JSON error] {e}", file=sys.stderr)


def robust_extract_content(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    try:
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                for msg in reversed(messages):
                    is_tool_msg = (
                        hasattr(msg, "__class__")
                        and msg.__class__.__name__ == "ToolMessage"
                    ) or (isinstance(msg, dict) and msg.get("type") == "tool")
                    if is_tool_msg:
                        continue
                    has_tool_calls = False
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        has_tool_calls = True
                    elif isinstance(msg, dict) and msg.get("tool_calls"):
                        has_tool_calls = True
                    if not has_tool_calls:
                        if hasattr(msg, "content") and msg.content:
                            return str(msg.content)
                        if (
                            isinstance(msg, dict)
                            and "content" in msg
                            and msg["content"]
                        ):
                            return str(msg["content"])
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    return "\n".join(text_parts)
        if isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                ch = response["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and "content" in ch["message"]:
                        return ch["message"]["content"]
                    if "text" in ch:
                        return ch["text"]
            if "content" in response:
                content = response["content"]
                if isinstance(content, str) and content:
                    return content
        if hasattr(response, "output"):
            output = response.output
            if isinstance(output, str):
                return output
            return robust_extract_content(output)
        result = str(response)
        if ("'messages':" in result or '"messages":' in result) and len(result) > 200:
            all_contents = re.findall(
                r"content=['\"]([^'\"]*(?:\\['\"][^'\"]*)*)['\"]", result
            )
            if all_contents:
                for content in reversed(all_contents):
                    if (
                        content
                        and content.strip()
                        and not content.startswith("HumanMessage")
                    ):
                        return content.replace("\\'", "'").replace('\\"', '"')
        return result
    except Exception as e:
        print(f"[robust_extract_content error] {e}", file=sys.stderr)
        return str(response)


def format_car_card(c: Dict[str, Any]) -> str:
    if not c:
        return "Unknown vehicle"
    make = sanitize_text(str(c.get("make", "") or "Unknown"))
    model = sanitize_text(str(c.get("model", "") or ""))
    year = str(c.get("year", "") or "")
    price = c.get("price")
    if price is None or price == "":
        price_str = "Price N/A"
    else:
        try:
            if isinstance(price, (int, float)) and float(price).is_integer():
                price_str = f"${int(price):,}"
            else:
                price_str = f"${float(price):,}"
        except Exception:
            price_str = str(price)
    mileage = c.get("mileage")
    mileage_str = f"{mileage} km" if mileage not in (None, "") else "Mileage N/A"
    desc = sanitize_text(str(c.get("description", "") or ""))
    if desc:
        desc = desc.split(".")[0][:100]
    title = " ".join(part for part in [make, model] if part).strip()
    if year:
        title = f"{title} ({year})"
    return f"{title} — {price_str} — {mileage_str}" + (f" — {desc}" if desc else "")


def build_results_message(cars: List[Dict[str, Any]]) -> str:
    if not cars:
        return "No cars matched your filters."
    total = len(cars)
    top = cars[0] if total > 0 else None
    lines = []
    for i, c in enumerate(cars[:8], start=1):
        lines.append(f"{i}. {format_car_card(c)}")
    best_text = ""
    if top:
        make = top.get("make", "")
        model = top.get("model", "")
        year = top.get("year", "")
        best_title = " ".join(part for part in [make, model] if part).strip()
        if year:
            best_title = f"{best_title} ({year})"
        best_text = f"Top pick: {best_title}."
    summary = f"I found {total} match{'es' if total != 1 else ''}. {best_text}\n"
    summary += "Reply with the number to select a car, or say 'more filters' to narrow results."
    return summary + "\n\n" + "\n".join(lines)


# ---------- Tools ----------
if LC_AVAILABLE:

    @tool("get_user_profile", description="Fetch user profile")
    def tool_get_user_profile(email: str) -> str:
        return fetch_user_profile_by_email(email)

    @tool("find_cars", description="Fetch cars in DB")
    def tool_find_cars(filters_json: str) -> str:
        try:
            filters = json.loads(filters_json) if isinstance(filters_json, str) else {}
        except Exception:
            filters = {"query": filters_json}
        cars = fetch_cars_by_filters(filters, limit=20)
        out = []
        for c in cars:
            c2 = {k: v for k, v in c.items() if k != "_id"}
            out.append(c2)
        if out:
            human_text = build_results_message(out)
            json_str = json.dumps(out, default=str)
            return f"{human_text}\n\n{CAR_JSON_MARKER}{json_str}"
        else:
            return "I couldn't find any cars matching those filters."

    @tool("web_search", description="Search the web")
    def tool_web_search(query: str) -> str:
        results = tavily_search_raw(query, max_results=3)
        human = "External search results:\n\n"
        lines = []
        for r in results:
            if isinstance(r, dict) and r.get("error"):
                lines.append(r.get("error"))
            else:
                title = r.get("title") or r.get("headline") or ""
                snippet = r.get("snippet") or r.get("summary") or ""
                url = r.get("url") or r.get("link") or ""
                lines.append(f"{title}\n{snippet}\n{url}")
        human += "\n\n".join(lines) if lines else str(results)
        return human + "\n\n" + WEB_JSON_MARKER + json.dumps(results, default=str)


@tool("place_order", description="Place order")
def tool_place_order(payload: str) -> str:
    import uuid

    print(f"[tool_place_order] Payload: {payload}", file=sys.stderr)
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except Exception as json_error:
        return f"Invalid JSON: {json_error}"

    def getd(d, *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
            if cur is default:
                return default
        return cur

    # --- Extract possible fields from many payload shapes ---
    incoming_session_id = (
        getd(data, "session_id")
        or getd(data, "session", "session_id")
        or getd(data, "session", "id")
    )
    buyer_email = (
        getd(data, "buyer_email")
        or getd(data, "email")
        or getd(data, "customer", "email")
        or getd(data, "session", "email")
    )
    buyer_name = (
        getd(data, "buyer_name")
        or getd(data, "customer_name")
        or getd(data, "customer", "name")
        or getd(data, "session", "customer_name")
    )
    buyer_phone = (
        getd(data, "buyer_phone")
        or getd(data, "phone")
        or getd(data, "customer", "phone")
        or getd(data, "session", "phone")
    )
    buyer_address = (
        getd(data, "buyer_address")
        or getd(data, "delivery_address")
        or getd(data, "address")
        or getd(data, "order", "delivery_address")
        or getd(data, "order", "address")
        or getd(data, "delivery", "address")
    )
    vehicle = (
        getd(data, "vehicle")
        or getd(data, "car")
        or getd(data, "order", "vehicle")
        or getd(data, "order", "car")
    )

    # If vehicle is a primitive string (e.g. "Porsche 911 (2020)") convert to dict
    if isinstance(vehicle, str):
        vehicle = {"description": vehicle}

    # Resolve authoritative session id:
    resolved_session_id = None
    # 1) If buyer_email is present, prefer the user's current_session/last_session in DB
    if buyer_email:
        try:
            u = users_col.find_one({"email": buyer_email})
            if u:
                # prefer current_session.session_id, then last_session_id
                cs = u.get("current_session", {}) or {}
                resolved_session_id = cs.get("session_id") or u.get("last_session_id")
        except Exception as e:
            print(f"[tool_place_order] user lookup error: {e}", file=sys.stderr)

    # 2) If not resolved yet, but incoming_session_id exists and is loaded in memory, prefer it
    if not resolved_session_id and incoming_session_id:
        if incoming_session_id in memory_manager.sessions:
            resolved_session_id = incoming_session_id
        else:
            # try to hydrate from DB
            try:
                memory_manager.ensure_session_loaded(
                    incoming_session_id, buyer_email or ""
                )
                if incoming_session_id in memory_manager.sessions:
                    resolved_session_id = incoming_session_id
            except Exception:
                pass

    # 3) If still not resolved, if incoming_session_id exists we will use it (create new session)
    if not resolved_session_id and incoming_session_id:
        resolved_session_id = incoming_session_id

    # 4) Last-resort: generate a canonical session id based on email or uuid
    if not resolved_session_id:
        if buyer_email:
            resolved_session_id = f"{buyer_email}_{int(time.time())}"
        else:
            resolved_session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Ensure session exists in memory and is persisted
    if resolved_session_id not in memory_manager.sessions:
        print(
            f"[tool_place_order] Creating minimal session: {resolved_session_id}",
            file=sys.stderr,
        )
        s = memory_manager._new_session(buyer_email or "")
        # keep email in the session
        s["user_email"] = buyer_email or s.get("user_email", "")
        memory_manager.sessions[resolved_session_id] = s
        try:
            persist_session_state_raw(
                s.get("user_email", buyer_email) or buyer_email or "",
                resolved_session_id,
                s,
            )
        except Exception as e:
            print(
                f"[tool_place_order] persist new session failed: {e}", file=sys.stderr
            )

    # Rehydrate data into session if available
    session = memory_manager.sessions.get(resolved_session_id)
    if session:
        # if incoming payload includes collected fields, store them in session.collected
        collected = session.get("collected", {}) or {}
        if buyer_name:
            collected["name"] = buyer_name
        if buyer_email:
            collected["email"] = buyer_email
        if buyer_phone:
            collected["phone"] = buyer_phone
        if buyer_address:
            collected["address"] = buyer_address
        session["collected"] = collected
        # If payload includes a vehicle object, set as selected_vehicle
        if vehicle:
            try:
                # ensure no ObjectId inside
                sel_copy = (
                    {k: (str(v) if k == "_id" else v) for k, v in vehicle.items()}
                    if isinstance(vehicle, dict)
                    else {"description": str(vehicle)}
                )
            except Exception:
                sel_copy = _make_json_safe(vehicle)
            session["selected_vehicle"] = sel_copy
            session["stage"] = "vehicle_selected"
        persist_session_state(resolved_session_id)

    # Ensure we have required fields for create_order_with_address
    if not buyer_address:
        # try session collected address as fallback
        buyer_address = session.get("collected", {}).get("address")

    if not session.get("selected_vehicle"):
        # As fallback, try to find last_results or selected_vehicle in DB user current_session
        print(
            f"[tool_place_order] No selected_vehicle in session {resolved_session_id}",
            file=sys.stderr,
        )
        # Let the caller know we need selection
        return "No vehicle selected in this session. Please select a vehicle before placing an order."

    if not buyer_address:
        return "Address required to place order. Provide 'buyer_address' or 'delivery_address'."

    # Prepare sales_contact fallback
    sales_contact = {
        "name": "Jeni Flemin",
        "position": "CEO",
        "phone": "+94778540035",
        "address": "Convent Garden, London, UK",
    }

    # Now call create_order_with_address safely
    try:
        order_id = create_order_with_address(
            session_id=resolved_session_id,
            buyer_name=buyer_name
            or session.get("user_email")
            or session.get("collected", {}).get("name"),
            vehicle=session.get("selected_vehicle"),
            sales_contact=sales_contact,
            buyer_address=buyer_address,
            buyer_phone=buyer_phone or session.get("collected", {}).get("phone"),
            buyer_email=buyer_email
            or session.get("collected", {}).get("email")
            or session.get("user_email"),
        )
        if order_id:
            # Update user's current_session in DB to reflect order
            try:
                users_col.update_one(
                    {"email": session.get("user_email")},
                    {
                        "$set": {
                            "current_session.session_id": resolved_session_id,
                            "current_session.stage": "finished",
                            "current_session.order_id": order_id,
                        }
                    },
                    upsert=True,
                )
            except Exception as e:
                print(
                    f"[tool_place_order] Failed updating user session in DB: {e}",
                    file=sys.stderr,
                )

            success_msg = f"✅ Order placed! ID: {order_id}\nVehicle: {session.get('selected_vehicle',{}).get('make') or session.get('selected_vehicle',{}).get('description','(desc)')}\nDelivery to: {buyer_address}"
            print(
                f"[tool_place_order] Order created: {order_id} (session {resolved_session_id})",
                file=sys.stderr,
            )
            return success_msg
        else:
            return "Failed to place order"
    except ValueError as ve:
        return f"Validation error: {ve}"
    except Exception as e:
        import traceback

        traceback.print_exc(file=sys.stderr)
        # Log failure doc
        try:
            failed_writes_col.insert_one(
                {
                    "collection": "orders",
                    "error": str(e),
                    "doc": {
                        "session_id": resolved_session_id,
                        "vehicle": session.get("selected_vehicle"),
                        "buyer_address": buyer_address,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": resolved_session_id,
                }
            )
        except Exception as log_error:
            print(
                f"[tool_place_order] Failed to log failure: {log_error}",
                file=sys.stderr,
            )
        return f"Error placing order: {e}"


# ---------- Create agents ----------
personal_agent = None
car_agent = None
web_agent = None
supervisor_agent = None

if LC_AVAILABLE:
    personal_prompt = "You are a Personal Agent. Fetch user profile when needed."
    personal_agent = create_agent(
        model=llm,
        name="PersonalAgent",
        system_prompt=personal_prompt,
        tools=[tool_get_user_profile],
    )

    car_prompt = (
        "You are a Car Sales Agent. Your job: help the user find and buy a car. Be concise and follow these rules strictly.\n\n"
        "1) Searching / showing cars:\n"
        "   - When asked to search, return human-friendly results and also include the exact JSON array of objects under the marker ===CAR_JSON=== so the system can store 'last_results'. Each object must include the car's fields (make, model, year, price, mileage, style, fuel_type, description, _id if available).\n\n"
        "2) Selection:\n"
        "   - When user replies with a number, interpret it as selecting that index from last shown ===CAR_JSON=== results. Confirm selection to user and store the full selected vehicle object in session memory.\n\n"
        "3) Order placement (VERY IMPORTANT):\n"
        "   - Before placing an order, ensure you have the following buyer details: buyer_name, buyer_email, buyer_phone, buyer_address.\n"
        "   - If any of those are missing in session memory, ask the user directly (one question at a time) to provide them. Do NOT call the place_order tool until all are present.\n"
        "   - Use the session's full selected_vehicle object (not a text description). If session only has a description, call out that you need to confirm the full vehicle (make/model/year/price) and ask the user or re-fetch results.\n"
        "   - When ready to place the order, **call the tool** with a single canonical JSON payload (top-level object) with these keys: \n"
        '     { "buyer_name": "...", "buyer_email": "...", "buyer_phone": "...", "buyer_address": "...", "vehicle": { ...full vehicle object... }}\n'
        "4) Confirmations:\n"
        "   - After calling place_order, present the user a short confirmation message including order id returned by the tool.\n\n"
        "Keep responses simple and actionable."
    )

    car_agent = create_agent(
        model=llm,
        name="CarSalesAgent",
        system_prompt=car_prompt,
        tools=[tool_find_cars, tool_place_order],
    )

    web_prompt = "You are a Web Agent. Search external sources."
    web_agent = create_agent(
        model=llm, name="WebAgent", system_prompt=web_prompt, tools=[tool_web_search]
    )

    supervisor_system_prompt = (
        "You are the Supervisor Agent. Be SIMPLE, DIRECT, and STATEFUL.\n\n"
        "Core Rules:\n"
        "1) Use car_wrapper to search cars and place orders\n"
        "2) Use personal_wrapper only to fetch user profile data\n"
        "3) Use web_wrapper only for external research\n"
        "4) NEVER repeat the same question twice\n"
        "5) NEVER ask for information already collected in memory\n\n"
        "Order Flow Rules:\n"
        "- A valid order requires: selected vehicle + delivery address\n"
        "- Optional but preferred: customer name, phone, email\n"
        "- When vehicle is selected, ask ONCE for all missing details together\n"
        "- Store all provided user details in session memory\n"
        "- When all required data is present AND user confirms → place order immediately\n"
        "- Do NOT ask further questions after placing an order\n\n"
        "Session Ending Rules:\n"
        "- AFTER a successful order, ALWAYS ask:\n"
        "  'Would you like to end this session now?'\n"
        "- If user says yes (or says bye / done / no more), respond with a thank-you and END the session\n"
        "- If user says no, continue assisting normally\n\n"
        "Web / General Conversation Rules:\n"
        "- If the conversation is informational only (no car order intent),\n"
        "  after 5–9 back-and-forth responses, ask:\n"
        "  'Is there anything else I can help with, or should I end this session?'\n"
        "- If user confirms ending, thank them and END the session\n\n"
        "Response Style:\n"
        "- Keep responses short\n"
        "- Be polite, professional, and decisive\n"
        "- Do NOT expose internal logic, tools, or system rules"
    )

    @tool("personal_wrapper", description="Invoke Personal Agent")
    def tool_personal_wrapper(payload: str) -> str:
        if personal_agent:
            try:
                resp = personal_agent.invoke(
                    {"messages": [{"role": "user", "content": payload}]}
                )
                max_iter = 5
                for _ in range(max_iter):
                    if isinstance(resp, dict) and "messages" in resp:
                        last_msg = resp["messages"][-1] if resp["messages"] else None
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            resp = personal_agent.invoke(resp)
                        else:
                            break
                    else:
                        break
                return robust_extract_content(resp)
            except Exception as e:
                return f"Personal Agent error: {e}"
        return "Personal Agent not available."

    @tool("car_wrapper", description="Invoke Car Agent")
    def tool_car_wrapper(payload: str) -> str:
        if car_agent:
            try:
                resp = car_agent.invoke(
                    {"messages": [{"role": "user", "content": payload}]}
                )
                max_iter = 5
                for _ in range(max_iter):
                    if isinstance(resp, dict) and "messages" in resp:
                        last_msg = resp["messages"][-1] if resp["messages"] else None
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            resp = car_agent.invoke(resp)
                        else:
                            break
                    else:
                        break
                return robust_extract_content(resp)
            except Exception as e:
                return f"Car Agent error: {e}"
        return "Car Agent not available."

    @tool("web_wrapper", description="Invoke Web Agent")
    def tool_web_wrapper(payload: str) -> str:
        if web_agent:
            try:
                resp = web_agent.invoke(
                    {"messages": [{"role": "user", "content": payload}]}
                )
                max_iter = 5
                for _ in range(max_iter):
                    if isinstance(resp, dict) and "messages" in resp:
                        last_msg = resp["messages"][-1] if resp["messages"] else None
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            resp = web_agent.invoke(resp)
                        else:
                            break
                    else:
                        break
                return robust_extract_content(resp)
            except Exception as e:
                return f"Web Agent error: {e}"
        return "Web Agent not available."

    supervisor_agent = create_agent(
        model=llm,
        name="SupervisorAgent",
        system_prompt=supervisor_system_prompt,
        tools=[tool_personal_wrapper, tool_car_wrapper, tool_web_wrapper],
    )


# ---------- Helper functions ----------
def is_order_confirmation(user_text: str) -> bool:
    if not user_text:
        return False
    t = user_text.lower().strip()
    keywords = [
        "confirm",
        "place order",
        "buy",
        "purchase",
        "i want this",
        "proceed",
        "yes i want",
        "go ahead",
        "yes",
    ]
    return any(k in t for k in keywords)


def contains_address_info(user_text: str) -> bool:
    if not user_text:
        return False
    t = user_text.lower()
    address_indicators = [
        "address",
        "street",
        "road",
        "avenue",
        "city",
        "phone",
        "email",
        "name:",
    ]
    return any(indicator in t for indicator in address_indicators)


def handle_car_selection(session_id: str, user_text: str) -> Optional[str]:
    s = memory_manager.sessions.get(session_id)
    if not s:
        return None
    if not user_text or not user_text.strip().isdigit():
        return None
    if not s.get("last_results"):
        return None
    try:
        idx = int(user_text.strip()) - 1
        if idx < 0 or idx >= len(s["last_results"]):
            return f"Selection {user_text.strip()} is out of range. Please choose between 1 and {len(s['last_results'])}."
        sel = s["last_results"][idx]
        try:
            sel_copy = {k: (str(v) if k == "_id" else v) for k, v in sel.items()}
        except Exception:
            sel_copy = _make_json_safe(sel)
        s["selected_vehicle"] = sel_copy
        s["stage"] = "vehicle_selected"
        s["awaiting"] = "address"
        persist_session_state(session_id)
        response = (
            f"✓ Great choice! You've selected:\n\n"
            f"🚗 {sel_copy.get('make')} {sel_copy.get('model')} ({sel_copy.get('year')})\n"
            f"💰 Price: ${sel_copy.get('price'):,}\n"
            f"📏 Mileage: {sel_copy.get('mileage')} km\n\n"
            f"To complete your order, please provide:\n"
            f"• Your full name\n"
            f"• Delivery address\n"
            f"• Phone number\n"
            f"• Email address\n\n"
            f"You can provide them all at once or one at a time."
        )
        return response
    except Exception as e:
        print("[handle_car_selection]", e, file=sys.stderr)
        return None


# ---------- Main supervisor invoke ----------
def supervisor_invoke(
    session_id: str, user_email: str, user_query: str
) -> Tuple[str, str]:
    session = memory_manager.sessions.get(session_id)
    if not session:
        memory_manager.ensure_session_loaded(session_id, user_email or "")
        session = memory_manager.sessions.get(session_id)
        if not session:
            # create and persist a new one (or raise)
            session = memory_manager._new_session(user_email or "")
            memory_manager.sessions[session_id] = session
            persist_session_state_raw(
                session.get("user_email", user_email) or user_email, session_id, session
            )

    # Extract contact info
    contact_info = extract_contact_info(user_query)
    if contact_info:
        print(f"[supervisor_invoke] Extracted: {contact_info}", file=sys.stderr)
        collected = session.get("collected", {})
        collected.update(contact_info)
        session["collected"] = collected
        persist_session_state(session_id)

    awaiting = session.get("awaiting")
    selected_vehicle = session.get("selected_vehicle")
    already_ordered = bool(session.get("order_id"))

    # Order placement logic
    if selected_vehicle and not already_ordered and is_order_confirmation(user_query):
        print(f"[supervisor_invoke] Order confirmation detected", file=sys.stderr)
        collected = session.get("collected", {})
        buyer_address = collected.get("address")

        if buyer_address:
            print(f"[supervisor_invoke] Have address, placing order", file=sys.stderr)
            try:
                oid = create_order_with_address(
                    session_id=session_id,
                    buyer_name=collected.get("name") or user_email,
                    vehicle=selected_vehicle,
                    sales_contact={
                        "name": "Jeni Flemin",
                        "position": "CEO",
                        "phone": "+94778540035",
                        "address": "Convent Garden, London, UK",
                    },
                    buyer_address=buyer_address,
                    buyer_phone=collected.get("phone"),
                    buyer_email=collected.get("email") or user_email,
                )

                if oid:
                    session["awaiting"] = None
                    persist_session_state(session_id)
                    success_msg = (
                        f"✅ Order placed successfully!\n\n"
                        f"Order ID: {oid}\n"
                        f"Vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})\n"
                        f"Price: ${selected_vehicle.get('price'):,}\n"
                        f"Delivery to: {buyer_address}\n\n"
                        f"Our team will contact you within 24 hours."
                    )
                    memory_manager.add_message(
                        session_id, user_query, success_msg, agent_used="OrderHandler"
                    )
                    return success_msg, session_id
            except Exception as e:
                error_msg = f"Sorry, there was an error: {str(e)}. Please try again."
                print(f"[supervisor_invoke] Order error: {e}", file=sys.stderr)
                memory_manager.add_message(
                    session_id, user_query, error_msg, agent_used="OrderHandler"
                )
                return error_msg, session_id
        else:
            session["awaiting"] = "address"
            persist_session_state(session_id)
            ask_text = (
                f"To complete your order for the {selected_vehicle.get('make')} {selected_vehicle.get('model')}, "
                f"I just need your delivery address.\n\n"
                f"Please provide your full address."
            )
            memory_manager.add_message(
                session_id, user_query, ask_text, agent_used="OrderHandler"
            )
            return ask_text, session_id

    # If awaiting address and user provides it
    if (
        awaiting == "address"
        and contains_address_info(user_query)
        and selected_vehicle
        and not already_ordered
    ):
        print(f"[supervisor_invoke] Address provided", file=sys.stderr)
        collected = session.get("collected", {})
        buyer_address = collected.get("address")

        if buyer_address:
            try:
                oid = create_order_with_address(
                    session_id=session_id,
                    buyer_name=collected.get("name") or user_email,
                    vehicle=selected_vehicle,
                    sales_contact={
                        "name": "Jeni Flemin",
                        "position": "CEO",
                        "phone": "+94778540035",
                        "address": "Convent Garden, London, UK",
                    },
                    buyer_address=buyer_address,
                    buyer_phone=collected.get("phone"),
                    buyer_email=collected.get("email") or user_email,
                )

                if oid:
                    session["awaiting"] = None
                    persist_session_state(session_id)
                    success_msg = (
                        f"✅ Order placed successfully!\n\n"
                        f"Order ID: {oid}\n"
                        f"Vehicle: {selected_vehicle.get('make')} {selected_vehicle.get('model')} ({selected_vehicle.get('year')})\n"
                        f"Price: ${selected_vehicle.get('price'):,}\n"
                        f"Delivery to: {buyer_address}\n\n"
                        f"Our team will contact you within 24 hours."
                    )
                    memory_manager.add_message(
                        session_id, user_query, success_msg, agent_used="OrderHandler"
                    )
                    return success_msg, session_id
            except Exception as e:
                error_msg = f"Sorry, there was an error: {str(e)}. Please try again."
                print(f"[supervisor_invoke] Order error: {e}", file=sys.stderr)
                memory_manager.add_message(
                    session_id, user_query, error_msg, agent_used="OrderHandler"
                )
                return error_msg, session_id

    # Handle numeric selection
    sel_reply = handle_car_selection(session_id, user_query)
    if sel_reply is not None:
        memory_manager.add_message(
            session_id, user_query, sel_reply, agent_used="SelectionHandler"
        )
        return sel_reply, session_id

    # Build context
    conversation_context = (
        memory_manager.get_context_for_llm(session_id)
        if hasattr(memory_manager, "get_context_for_llm")
        else None
    )

    state_context = ""
    if selected_vehicle and not already_ordered:
        state_context = f"\n[Selected: {selected_vehicle.get('make')} {selected_vehicle.get('model')} - ${selected_vehicle.get('price'):,}]"
        collected = session.get("collected", {})
        if collected.get("address"):
            state_context += f"\n[Have address: {collected.get('address')}]"
            state_context += "\n[READY TO ORDER - User just needs to confirm]"
        else:
            state_context += "\n[Need address]"

    full_prompt = (
        f"\nPrevious:\n{conversation_context}\n{state_context}\n\nUser: {user_query}\n"
        if conversation_context
        else user_query
    )

    # Get response from supervisor
    if LC_AVAILABLE and supervisor_agent:
        try:
            messages = [{"role": "user", "content": full_prompt}]
            resp = supervisor_agent.invoke({"messages": messages})

            max_iterations = 10
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                if isinstance(resp, dict) and "messages" in resp:
                    messages_list = resp["messages"]
                    if not messages_list:
                        break
                    last_msg = messages_list[-1]
                    has_tool_calls = False
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        has_tool_calls = True
                    elif isinstance(last_msg, dict) and last_msg.get("tool_calls"):
                        has_tool_calls = True
                    is_tool_message = (
                        hasattr(last_msg, "__class__")
                        and last_msg.__class__.__name__ == "ToolMessage"
                    ) or (isinstance(last_msg, dict) and last_msg.get("type") == "tool")
                    if has_tool_calls or is_tool_message:
                        try:
                            resp = supervisor_agent.invoke(resp)
                        except Exception as e:
                            print(f"[agent error] {e}", file=sys.stderr)
                            break
                    else:
                        break
                else:
                    break
            out_raw = robust_extract_content(resp)
        except Exception as e:
            print(f"[supervisor error] {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            out_raw = f"Sorry, I encountered an error: {e}"
    else:
        try:
            resp = llm([{"role": "user", "content": full_prompt}])
            out_raw = robust_extract_content(resp)
        except Exception as e:
            out_raw = f"Fallback: {user_query} (error: {e})"

    # Extract JSON markers
    try:
        extract_and_store_json_markers_safe(str(out_raw), session_id, memory_manager)
    except Exception as e:
        print("[extract error]", e, file=sys.stderr)

    # Clean output
    cleaned_output = out_raw
    if CAR_JSON_MARKER in cleaned_output:
        cleaned_output = cleaned_output.split(CAR_JSON_MARKER)[0]
    if WEB_JSON_MARKER in cleaned_output:
        cleaned_output = cleaned_output.split(WEB_JSON_MARKER)[0]
    cleaned_output = cleaned_output.strip()

    # Save to history
    memory_manager.add_message(
        session_id, user_query, cleaned_output, agent_used="Supervisor"
    )
    return cleaned_output, session_id


# ---------- Top-level API ----------
QUESTION_LIMIT = 6


def handle_user_query(
    session_id: Optional[str], user_email: str, user_query: str
) -> Dict[str, Any]:

    sid = memory_manager.get_or_create_session(user_email, session_id)
    memory_manager.sessions[sid]["user_email"] = user_email

    # Normal response
    resp, _ = supervisor_invoke(sid, user_email, user_query)

    # Count user questions
    message_count = len(memory_manager.sessions[sid]["messages"])

    # ✅ Auto end after 6 questions
    if message_count >= QUESTION_LIMIT:
        summary = memory_manager.end_session_and_save(sid)

        return {
            "response": resp,
            "session_id": sid,
            "session_ended": True,
            "conversation_summary": summary,
            "message": "Conversation ended after 6 questions",
        }

    return {"response": resp, "session_id": sid, "session_ended": False}


def end_session(session_id: str, user_email: str) -> dict:
    session = memory_manager.sessions.get(session_id)

    if not session:
        return {"status": "error", "message": "Session not found or already ended"}

    summary = memory_manager.generate_summary(session_id)

    memory_manager.end_session_and_save(session_id)

    return {
        "status": "ended",
        "session_id": session_id,
        "user_email": user_email,
        "conversation_summary": summary,
    }


__all__ = [
    "handle_user_query",
    "end_session",
    "memory_manager",
    "create_order_with_address",
    "persist_session_state",
]
