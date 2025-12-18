import sys
import re
from typing import Dict, Any, List
from database import users_col, cars_col
from utils import utcnow_iso, _make_json_safe, sanitize_text
from config import TAVILY_API_KEY


def persist_session_state(session_id: str):
    """Persist session state to database"""
    from conversation_memory import memory_manager

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
    """Persist session state to database (raw version)"""
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
    """Fetch user profile from database"""
    if not email:
        return "No email provided."
    p = users_col.find_one({"email": email})
    if not p:
        return f"No profile found for {email}."
    return f"Name: {p.get('name','')}\nEmail: {p.get('email','')}\nRecent summary: {p.get('recent_summary')}"


def fetch_cars_by_filters(
    filters: Dict[str, Any], limit: int = 10
) -> List[Dict[str, Any]]:
    """Fetch cars from database with filters"""
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
    """Search the web using Tavily API"""
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


def format_car_card(c: Dict[str, Any]) -> str:
    """Format car data as a readable string"""
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
    """Build a user-friendly message showing search results"""
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


def is_order_confirmation(user_text: str) -> bool:
    """Check if user is confirming an order"""
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
    """Check if text contains address information"""
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


def handle_car_selection(session_id: str, user_text: str):
    """Handle numeric car selection from search results"""
    from conversation_memory import memory_manager
    from utils import _make_json_safe

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
