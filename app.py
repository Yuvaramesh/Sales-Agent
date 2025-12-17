"""
Flask Web Application for Car Recommendation Chatbot
"""

from flask import Flask, render_template, request, jsonify, session
import requests
import os
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# Backend API URL (adjust based on your setup)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@app.route("/")
def index():
    """Render the main chat interface"""
    # Generate a new session if one doesn't exist
    if "user_session_id" not in session:
        session["user_session_id"] = None
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_email = data.get("user_email")
        user_query = data.get("message")
        session_id = session.get("user_session_id")

        if not user_email or not user_query:
            return jsonify({"error": "Missing required fields"}), 400

        # Call backend API
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "session_id": session_id,
                "user_email": user_email,
                "user_query": user_query,
            },
            timeout=60,
        )

        if response.status_code == 200:
            result = response.json()
            # Store session_id for future requests
            session["user_session_id"] = result.get("session_id")
            return jsonify(
                {
                    "response": result.get("response"),
                    "session_id": result.get("session_id"),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        else:
            return jsonify({"error": "Backend error"}), 500

    except requests.Timeout:
        return jsonify({"error": "Request timeout"}), 504
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/end-session", methods=["POST"])
def end_session_route():
    """End the current chat session"""
    try:
        data = request.json
        user_email = data.get("user_email")
        session_id = session.get("user_session_id")

        if not session_id:
            return jsonify({"message": "No active session"}), 200

        # Call backend API
        response = requests.post(
            f"{BACKEND_URL}/end_session",
            json={"session_id": session_id, "user_email": user_email},
            timeout=30,
        )

        # Clear session
        session.pop("user_session_id", None)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to end session"}), 500

    except Exception as e:
        print(f"Error ending session: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "flask_ui"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
