import sys
from utils import sanitize_text, estimate_tokens, utcnow_iso, robust_extract_content
from llm_setup import llm


class MemoryOptimizerMixin:
    """Mixin for managing conversation memory with summarization"""

    MAX_PROMPT_TOKENS = 3000
    RECENT_TURNS_KEEP = 8
    SUMMARIZE_EVERY = 12
    SUMMARY_MAX_TOKENS = 800

    def compress_history_if_needed(self, session_id: str):
        """Compress older conversation history into summaries"""
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
            from helpers import persist_session_state

            persist_session_state(session_id)
        except Exception as e:
            print("[compress_history persist error]", e, file=sys.stderr)

    def get_context_for_llm(self, session_id: str, max_messages: int = None) -> str:
        """Build context string for LLM from session history"""
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
