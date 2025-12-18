import json
import sys
import time
import uuid
from typing import Optional
from datetime import datetime, timezone
from llm_setup import LC_AVAILABLE
from helpers import (
    fetch_user_profile_by_email,
    fetch_cars_by_filters,
    tavily_search_raw,
    build_results_message,
    persist_session_state,
)
from order_helpers import create_order_with_address
from config import CAR_JSON_MARKER, WEB_JSON_MARKER
from database import users_col, failed_writes_col

if LC_AVAILABLE:
    from langchain.tools import tool

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
    """Tool for placing vehicle orders"""
    from conversation_memory import memory_manager
    from utils import _make_json_safe

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

    # Extract fields from various payload shapes
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

    # Convert string vehicle to dict
    if isinstance(vehicle, str):
        vehicle = {"description": vehicle}

    # Resolve session ID
    resolved_session_id = None
    if buyer_email:
        try:
            u = users_col.find_one({"email": buyer_email})
            if u:
                cs = u.get("current_session", {}) or {}
                resolved_session_id = cs.get("session_id") or u.get("last_session_id")
        except Exception as e:
            print(f"[tool_place_order] user lookup error: {e}", file=sys.stderr)

    if not resolved_session_id and incoming_session_id:
        if incoming_session_id in memory_manager.sessions:
            resolved_session_id = incoming_session_id
        else:
            try:
                memory_manager.ensure_session_loaded(
                    incoming_session_id, buyer_email or ""
                )
                if incoming_session_id in memory_manager.sessions:
                    resolved_session_id = incoming_session_id
            except Exception:
                pass

    if not resolved_session_id and incoming_session_id:
        resolved_session_id = incoming_session_id

    if not resolved_session_id:
        if buyer_email:
            resolved_session_id = f"{buyer_email}_{int(time.time())}"
        else:
            resolved_session_id = f"sess_{uuid.uuid4().hex[:8]}"

    # Ensure session exists
    if resolved_session_id not in memory_manager.sessions:
        print(
            f"[tool_place_order] Creating minimal session: {resolved_session_id}",
            file=sys.stderr,
        )
        s = memory_manager._new_session(buyer_email or "")
        s["user_email"] = buyer_email or s.get("user_email", "")
        memory_manager.sessions[resolved_session_id] = s
        try:
            from helpers import persist_session_state_raw

            persist_session_state_raw(
                s.get("user_email", buyer_email) or buyer_email or "",
                resolved_session_id,
                s,
            )
        except Exception as e:
            print(
                f"[tool_place_order] persist new session failed: {e}", file=sys.stderr
            )

    # Update session with collected data
    session = memory_manager.sessions.get(resolved_session_id)
    if session:
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

        if vehicle:
            try:
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

    # Validate requirements
    if not buyer_address:
        buyer_address = session.get("collected", {}).get("address")

    if not session.get("selected_vehicle"):
        print(
            f"[tool_place_order] No selected_vehicle in session {resolved_session_id}",
            file=sys.stderr,
        )
        return "No vehicle selected in this session. Please select a vehicle before placing an order."

    if not buyer_address:
        return "Address required to place order. Provide 'buyer_address' or 'delivery_address'."

    # Create order
    sales_contact = {
        "name": "Jeni Flemin",
        "position": "CEO",
        "phone": "+94778540035",
        "address": "Convent Garden, London, UK",
    }

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
