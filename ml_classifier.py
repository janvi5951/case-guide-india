import pickle
import os

# ============================================================
# ml_classifier.py — The CLASSIFIER worker 🤖
# Place this file in your backend/ folder
#
# HOW IT WORKS:
# 1. Loads the saved ML model on startup
# 2. classify(text) — pass any query → get category back
# ============================================================

# ── Load model when server starts ────────────────────────────
MODEL_PATH      = "ml_model/classifier.pkl"
VECTORIZER_PATH = "ml_model/vectorizer.pkl"

model     = None
vectorizer = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ ML Classifier loaded successfully!")

except FileNotFoundError:
    print("⚠️  ML model not found. Run train_model.py first!")
except Exception as e:
    print(f"⚠️  ML model load error: {e}")


def classify(query_text):
    """
    Takes a user's query → returns predicted category + confidence

    Example:
        classify("my phone was stolen")
        → { "category": "FIR", "confidence": 62 }

    Example:
        classify("online fraud happened")
        → { "category": "Cybercrime", "confidence": 85 }
    """
    if model is None or vectorizer is None:
        return {
            "category":   "General",
            "confidence": 0,
            "error":      "Model not loaded. Run train_model.py first."
        }

    if not query_text or not query_text.strip():
        return { "category": "General", "confidence": 0 }

    # Convert text → numbers → predict
    vec        = vectorizer.transform([query_text.lower()])
    category   = model.predict(vec)[0]
    confidence = round(model.predict_proba(vec).max() * 100)

    return {
        "category":   category,
        "confidence": confidence
    }


def classify_and_explain(query_text):
    """
    Same as classify() but also returns a helpful message
    telling the user what to do next.

    Used by the chatbot to give a complete response.
    """
    result = classify(query_text)
    category = result["category"]

    # Helpful next-step message for each category
    next_steps = {
        "FIR":               "📋 Please visit your nearest police station to file an FIR. You can also report online at your state police website.",
        "Cybercrime":        "🚨 Call cybercrime helpline 1930 immediately or visit cybercrime.gov.in to report online.",
        "Domestic Violence": "💛 Call women helpline 1091 for immediate help or visit nearest women police station.",
        "Consumer":          "🛒 File your complaint online at edaakhil.nic.in or visit the District Consumer Forum.",
        "RTI":               "📄 File your RTI application at rtionline.gov.in with a fee of just Rs. 10.",
        "Property":          "🏠 Consult a civil lawyer or file complaint at the civil court. For builder fraud try RERA portal.",
        "Women Safety":      "👩 Call 112 for emergency or 1091 for women helpline. You can also file at nearest police station.",
        "Legal Aid":         "⚖️ Contact your District Legal Services Authority (DLSA) for free legal help. Call NALSA at 15100.",
        "Child Safety":      "🧒 Call Childline 1098 immediately — available 24/7 free of cost.",
        "General":           "ℹ️ Please visit our Guidance page for detailed help or contact NALSA helpline at 15100.",
    }

    result["message"] = next_steps.get(category, next_steps["General"])
    return result