import sys
import traceback
from typing import Dict, Any, Optional
from datetime import datetime, timezone, date
from bson import ObjectId
from database import orders_col, failed_writes_col


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
    from conversation_memory import memory_manager
    from helpers import persist_session_state

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
