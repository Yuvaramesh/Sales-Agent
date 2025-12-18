from typing import Dict, Any, Optional
from conversation_memory import memory_manager
from supervisor import supervisor_invoke
from config import QUESTION_LIMIT


def handle_user_query(
    session_id: Optional[str], user_email: str, user_query: str
) -> Dict[str, Any]:
    """
    Main API function to handle user queries

    Args:
        session_id: Optional session ID, will create new if None
        user_email: User's email address
        user_query: User's question/message

    Returns:
        Dictionary with response, session_id, and session status
    """
    sid = memory_manager.get_or_create_session(user_email, session_id)
    memory_manager.sessions[sid]["user_email"] = user_email

    # Get response from supervisor
    resp, _ = supervisor_invoke(sid, user_email, user_query)

    # Count user questions
    message_count = len(memory_manager.sessions[sid]["messages"])

    # Auto end after question limit
    if message_count >= QUESTION_LIMIT:
        summary = memory_manager.end_session_and_save(sid)

        return {
            "response": resp,
            "session_id": sid,
            "session_ended": True,
            "conversation_summary": summary,
            "message": f"Conversation ended after {QUESTION_LIMIT} questions",
        }

    return {"response": resp, "session_id": sid, "session_ended": False}


def end_session(session_id: str, user_email: str) -> Dict[str, Any]:
    """
    Manually end a session

    Args:
        session_id: Session ID to end
        user_email: User's email address

    Returns:
        Dictionary with session end status and summary
    """
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
]
