import sys
import traceback
from typing import Tuple
from conversation_memory import memory_manager
from llm_setup import LC_AVAILABLE, llm
from agents import supervisor_agent
from helpers import (
    is_order_confirmation,
    contains_address_info,
    handle_car_selection,
    persist_session_state,
)
from order_helpers import create_order_with_address
from utils import (
    robust_extract_content,
    extract_contact_info,
    extract_and_store_json_markers_safe,
)
from config import CAR_JSON_MARKER, WEB_JSON_MARKER


def supervisor_invoke(
    session_id: str, user_email: str, user_query: str
) -> Tuple[str, str]:
    """Main supervisor function to handle user queries"""
    session = memory_manager.sessions.get(session_id)
    if not session:
        memory_manager.ensure_session_loaded(session_id, user_email or "")
        session = memory_manager.sessions.get(session_id)
        if not session:
            # Create new session
            session = memory_manager._new_session(user_email or "")
            memory_manager.sessions[session_id] = session
            from helpers import persist_session_state_raw

            persist_session_state_raw(
                session.get("user_email", user_email) or user_email, session_id, session
            )

    # CHECK FOR SESSION END REQUEST - ADD THIS SECTION
    end_keywords = [
        "end the session",
        "end session",
        "end conversation",
        "end this session",
        "bye",
        "goodbye",
        "end now",
        "finish",
        "done",
        "close session",
    ]
    user_query_lower = user_query.lower().strip()

    if any(keyword in user_query_lower for keyword in end_keywords):
        # Generate and save summary immediately
        summary = memory_manager.end_session_and_save(session_id)

        goodbye_msg = (
            "Thank you for using AutoBot! 🚗\n\n"
            "Your session has been saved. We look forward to helping you again!\n\n"
            "Have a great day! 👋"
        )

        memory_manager.add_message(
            session_id, user_query, goodbye_msg, agent_used="SessionHandler"
        )
        return goodbye_msg, session_id

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
