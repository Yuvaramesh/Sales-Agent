import re
import json
import sys
from typing import Any, Dict
from datetime import datetime, timezone, date
from decimal import Decimal
from bson import ObjectId


def utcnow_iso() -> str:
    """Return current UTC time as ISO string"""
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(s: str, max_len: int = 4000) -> str:
    """Sanitize and truncate text"""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _make_json_safe(obj: Any) -> Any:
    """Convert object to MongoDB-safe format while preserving types"""
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
    """Ensure vehicle is a dict"""
    if not vehicle:
        return None
    if isinstance(vehicle, dict):
        return vehicle
    return None


def estimate_tokens(text: str) -> int:
    """Rough token estimation"""
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


def robust_extract_content(response) -> str:
    """Extract content from various response formats"""
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


def extract_and_store_json_markers_safe(text: str, session_id: str, memory_manager):
    """Extract JSON data from response markers"""
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

    from config import CAR_JSON_MARKER, WEB_JSON_MARKER

    if CAR_JSON_MARKER in text:
        try:
            after = text.split(CAR_JSON_MARKER, 1)[1]
            parsed = _parse_json_after_marker(after)
            if parsed is not None:
                s = memory_manager.sessions.setdefault(
                    session_id, memory_manager._new_session("")
                )
                s["last_results"] = parsed
                from helpers import persist_session_state

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
                from helpers import persist_session_state

                persist_session_state(session_id)
        except Exception as e:
            print(f"[extract WEB_JSON error] {e}", file=sys.stderr)
