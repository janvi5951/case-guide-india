from flask import Flask, jsonify, request, render_template, session
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq as GroqClient
import os
"""
If you see 'Import "dotenv" could not be resolved', install the package:
    pip install python-dotenv
"""
from dotenv import load_dotenv

# ============================================================
# LOAD .env — reads GROQ_API_KEY, DATABASE_URL, SECRET_KEY
# ============================================================
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY   = os.environ.get("SECRET_KEY", "caseguide_secret_fallback")

# Safe Groq client init — won't crash if key is missing
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Add it to your .env file or Render environment variables.")
groq_client        = GroqClient(api_key=GROQ_API_KEY)
conversation_store = {}

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

# ============================================================
# DATABASE CONNECTION — works on both localhost and Render
# ============================================================
def get_db():
    # On Render, DATABASE_URL is set in environment variables
    # On local, it reads from .env file
    db_url = DATABASE_URL
    if db_url:
        # Render gives postgres:// but psycopg2 needs postgresql://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(db_url)
    else:
        # Fallback to individual config values (local dev)
        return psycopg2.connect(
            dbname   = "case_guide_india",
            user     = "postgres",
            password = "janvi",
            host     = "localhost",
            port     = "5432"
        )

# ============================================================
# LOAD ML CLASSIFIER
# ============================================================
try:
    import pickle
    with open("ml_model/classifier.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open("ml_model/vectorizer.pkl", "rb") as f:
        ml_vectorizer = pickle.load(f)
    print("ML Classifier loaded!")
except Exception as e:
    print(f"ML model not found: {e}")
    ml_model      = None
    ml_vectorizer = None

# ============================================================
# LOAD CHATBOT CSV
# ============================================================
try:
    legal_data      = pd.read_csv("legal_queries.csv")
    questions_list  = legal_data["question"].tolist()
    answers_list    = legal_data["answer"].tolist()
    chat_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    chat_vectorizer.fit_transform(questions_list)
    print(f"Chatbot CSV loaded! {len(questions_list)} Q&A pairs ready.")
except Exception as e:
    print(f"Could not load legal_queries.csv: {e}")
    questions_list  = []
    answers_list    = []
    chat_vectorizer = TfidfVectorizer()

# ============================================================
# KEYWORD FALLBACK
# ============================================================
KEYWORD_ANSWERS = {
    "phone lost":        "If your phone is lost or stolen, file an FIR at your nearest police station. Also call your network provider to block your SIM. Report at cybercrime.gov.in if it is being misused.",
    "lost my phone":     "If your phone is lost or stolen, file an FIR at your nearest police station. Also call your network provider to block your SIM. Report at cybercrime.gov.in if it is being misused.",
    "phone stolen":      "If your phone is stolen, immediately file an FIR at your nearest police station and call your network provider to block your SIM card.",
    "mobile stolen":     "If your phone is stolen, immediately file an FIR at your nearest police station and call your network provider to block your SIM card.",
    "mobile lost":       "If your phone is lost, file an FIR at the nearest police station. You can also file an online complaint at your state police portal.",
    "fraud":             "Call cybercrime helpline 1930 immediately or visit cybercrime.gov.in to file an online complaint. Also inform your bank if money was deducted.",
    "online fraud":      "Call cybercrime helpline 1930 immediately or visit cybercrime.gov.in to file an online complaint. Also inform your bank if money was deducted.",
    "money deducted":    "Call cybercrime helpline 1930 immediately and inform your bank. Also file a complaint at cybercrime.gov.in.",
    "otp fraud":         "Call cybercrime helpline 1930 immediately. Also inform your bank to block your card and freeze your account.",
    "hacked":            "Call cybercrime helpline 1930 or visit cybercrime.gov.in to report hacking. Change all passwords immediately.",
    "blackmail":         "Call cybercrime helpline 1930 and file a complaint at cybercrime.gov.in. Do not pay any ransom. Save all evidence.",
    "husband beating":   "Call women helpline 1091 immediately. Visit the nearest women police station to file a complaint under the Domestic Violence Act.",
    "beating":           "Call women helpline 1091 immediately. Visit the nearest women police station to file a complaint under the Domestic Violence Act.",
    "domestic violence": "Call women helpline 1091 immediately. Visit the nearest women police station to file a complaint under the Domestic Violence Act.",
    "dowry":             "File a complaint at the nearest police station under Section 498A IPC. You can also call women helpline 1091.",
    "husband":           "Call women helpline 1091 for domestic violence issues. Visit the nearest women police station for help.",
    "fir":               "Visit your nearest police station to file an FIR. If police refuse, you can complain to the Superintendent of Police or file an online FIR at your state police website.",
    "police complaint":  "Visit your nearest police station to file a complaint. If police refuse to register FIR, approach the Superintendent of Police.",
    "stolen":            "File an FIR at your nearest police station immediately. Carry any proof of the stolen item such as purchase receipt or photos.",
    "theft":             "File an FIR at your nearest police station immediately. Carry any proof of the stolen item such as purchase receipt or photos.",
    "robbery":           "Call 112 for emergency and file an FIR at the nearest police station immediately.",
    "refund":            "File a consumer complaint at edaakhil.nic.in or visit your District Consumer Forum. You can claim refund plus compensation.",
    "consumer complaint":"File your complaint at edaakhil.nic.in or visit your District Consumer Forum. No lawyer needed for cases below Rs. 1 crore.",
    "rti":               "File your RTI application at rtionline.gov.in. Pay a fee of Rs. 10. The Public Information Officer must reply within 30 days.",
    "free lawyer":       "Contact your District Legal Services Authority (DLSA) for free legal help. Call NALSA helpline at 15100.",
    "legal aid":         "Call NALSA helpline at 15100 for free legal help. You can also visit your District Legal Services Authority (DLSA) office.",
    "stalking":          "Call 112 for emergency or 1091 for women helpline. File a complaint at nearest police station under Section 354D IPC.",
    "harassment":        "File a complaint at the nearest police station. For workplace harassment, file with your company's Internal Complaints Committee under the POSH Act.",
    "child":             "Call Childline 1098 immediately - available 24/7 and completely free for children in need of protection.",
    "childline":         "Call Childline 1098 immediately - available 24/7 and completely free for children in need of protection.",
    "help":              "For emergency call 112. Women helpline: 1091. Cybercrime: 1930. Childline: 1098. Legal Aid NALSA: 15100.",
    "emergency":         "For emergency call 112. Women helpline: 1091. Cybercrime: 1930. Childline: 1098. Legal Aid NALSA: 15100.",
    "helpline":          "Important helplines: Emergency 112, Women 1091, Cybercrime 1930, Childline 1098, Legal Aid NALSA 15100.",
}

def keyword_search(message):
    msg = message.lower()
    for keyword, answer in KEYWORD_ANSWERS.items():
        if keyword in msg:
            return answer
    return None

# ============================================================
# PAGE ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/guidance_page")
def guidance_page():
    return render_template("guidance.html")

@app.route("/query_page")
def query_page():
    return render_template("query.html")

@app.route("/feedback_page")
def feedback_page():
    return render_template("feedback.html")

@app.route("/chat_page")
def chat_page():
    return render_template("chat.html")

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT message FROM chat_messages WHERE sender = 'user'")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    topics = {"FIR": 0, "Cybercrime": 0, "Domestic Violence": 0, "Consumer": 0, "Other": 0}
    for row in rows:
        q = row[0].lower()
        if "fir" in q or "police" in q or "stolen" in q or "theft" in q:
            topics["FIR"] += 1
        elif "cyber" in q or "fraud" in q or "online" in q or "hack" in q:
            topics["Cybercrime"] += 1
        elif "violence" in q or "domestic" in q or "husband" in q or "beating" in q:
            topics["Domestic Violence"] += 1
        elif "consumer" in q or "refund" in q or "product" in q:
            topics["Consumer"] += 1
        else:
            topics["Other"] += 1

    return render_template("dashboard.html", topics=topics)

# ============================================================
# SIGNUP API
# ============================================================
@app.route("/signup", methods=["POST"])
def signup():
    data     = request.json
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"message": "Name, email and password are required"}), 400

    if len(password) < 6:
        return jsonify({"message": "Password must be at least 6 characters"}), 400

    hashed_password = generate_password_hash(password)
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id FROM app_users WHERE email = %s", (email,))
        if cur.fetchone():
            return jsonify({"message": "Email already registered. Please login."}), 409
        cur.execute(
            "INSERT INTO app_users (username, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        conn.commit()
        return jsonify({"message": "Account created successfully! Please login."}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"message": f"Error: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

# ============================================================
# LOGIN API
# ============================================================
@app.route("/login", methods=["POST"])
def login():
    data     = request.json
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"message": "Email and password are required"}), 400

    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id, username, password_hash FROM app_users WHERE email = %s",
            (email,)
        )
        user = cur.fetchone()
        if not user:
            return jsonify({"message": "No account found with this email"}), 404

        user_id, username, password_hash = user
        if check_password_hash(password_hash, password):
            session["user_id"]  = user_id
            session["username"] = username
            return jsonify({
                "message":  "Login successful!",
                "user_id":  user_id,
                "username": username
            }), 200
        else:
            return jsonify({"message": "Incorrect password. Please try again."}), 401
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

# ============================================================
# LOGOUT API
# ============================================================
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"})

# ============================================================
# GUIDANCE API
# ============================================================
@app.route("/guidance", methods=["GET"])
def get_guidance():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, case_type, description, steps, authority FROM case_guidance")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id":          row[0],
            "case_type":   row[1],
            "description": row[2],
            "steps":       row[3],
            "authority":   row[4]
        })
    return jsonify(result)

@app.route("/guidance/<category>", methods=["GET"])
def get_guidance_by_category(category):
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id, case_type, description, steps, authority FROM case_guidance WHERE LOWER(case_type) = LOWER(%s) LIMIT 1",
            (category,)
        )
        row = cur.fetchone()
        if row:
            return jsonify({
                "id":          row[0],
                "case_type":   row[1],
                "description": row[2],
                "steps":       row[3],
                "authority":   row[4]
            })
        return jsonify({"message": "No guidance found for this category"}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ============================================================
# QUERIES API
# ============================================================
@app.route("/queries", methods=["GET"])
def get_queries():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, user_name, query, category, created_at FROM queries")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id":         row[0],
            "user_name":  row[1],
            "query":      row[2],
            "category":   row[3],
            "created_at": str(row[4])
        })
    return jsonify(result)

@app.route("/submit_query", methods=["POST"])
def submit_query():
    data      = request.json
    user_name = data.get("user_name", "Anonymous").strip()
    query     = data.get("query", "").strip()
    category  = data.get("category", "").strip()
    gender    = data.get("gender", "").strip()

    if not query:
        return jsonify({"message": "Query is required"}), 400

    confidence = 0
    if not category:
        try:
            if ml_model and ml_vectorizer:
                vec        = ml_vectorizer.transform([query.lower()])
                category   = ml_model.predict(vec)[0]
                confidence = round(ml_model.predict_proba(vec).max() * 100)
            else:
                category = "General"
        except Exception:
            category = "General"
    else:
        try:
            if ml_model and ml_vectorizer:
                vec        = ml_vectorizer.transform([query.lower()])
                confidence = round(ml_model.predict_proba(vec).max() * 100)
        except Exception:
            confidence = 0

    gender_context = ""
    if gender == "Female":
        gender_context = "The person asking this question is a WOMAN. Prioritize women-specific laws like Protection of Women from Domestic Violence Act 2005, Section 498A IPC, POSH Act 2013, women helpline 1091, women police stations, and Mahila courts where applicable."
    elif gender == "Male":
        gender_context = "The person asking this question is a MAN. Give advice relevant to men, including situations where men may be falsely accused, or need legal protection as husbands/fathers/employees."
    else:
        gender_context = "Gender not specified. Give neutral, inclusive legal advice applicable to all."

    query_prompt = f"""You are Case Guide AI, an Indian legal assistant. Answer the following legal query in detail.

USER DETAILS:
- Name: {user_name}
- Gender: {gender if gender else 'Not specified'}
- Legal Category: {category}

GENDER CONTEXT:
{gender_context}

USER'S LEGAL QUERY:
"{query}"

INSTRUCTIONS:
1. One sentence of empathy only
2. One sentence naming the exact law/act that applies
3. Numbered steps — maximum 6 steps, each step maximum 1 sentence, start each with an action word
4. One line: relevant helpline number
5. One line: relevant online portal if available
6. Keep total response under 200 words
7. Plain text only — no markdown, no ** symbols, no long paragraphs"""

    answer = "Please visit your nearest legal authority or call NALSA helpline at 15100 for help."
    try:
        ai_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are Case Guide AI, an expert Indian legal assistant. Always give detailed, accurate, empathetic answers in plain text without markdown symbols."},
                {"role": "user", "content": query_prompt}
            ],
            temperature=0.6,
            max_tokens=800
        )
        answer = ai_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq query error: {e}")
        answer = "Our AI is temporarily unavailable. Please call NALSA helpline 15100 or visit your nearest District Legal Services Authority for free legal help."

    conn = get_db()
    cur  = conn.cursor()
    try:
        try:
            cur.execute(
                "INSERT INTO queries (user_name, query, category, confidence, status) VALUES (%s, %s, %s, %s, %s)",
                (user_name, query, category, confidence, "pending")
            )
        except Exception:
            conn.rollback()
            cur.execute(
                "INSERT INTO queries (user_name, query, category, status) VALUES (%s, %s, %s, %s)",
                (user_name, query, category, "pending")
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Query save error: {e}")
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "message":    "Query submitted successfully!",
        "answer":     answer,
        "category":   category,
        "confidence": confidence
    })

# ============================================================
# FEEDBACK API
# ============================================================
@app.route("/feedback", methods=["GET"])
def get_feedback():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT id, rating, comments, created_at FROM user_feedback")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for row in rows:
        result.append({
            "id":         row[0],
            "rating":     row[1],
            "comments":   row[2],
            "created_at": str(row[3])
        })
    return jsonify(result)

@app.route("/feedback", methods=["POST"])
def add_feedback():
    data     = request.json
    rating   = data.get("rating")
    comments = data.get("comments", "")

    if not rating:
        return jsonify({"message": "Rating is required"}), 400

    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO user_feedback (rating, comments) VALUES (%s, %s)",
            (rating, comments)
        )
        conn.commit()
        return jsonify({"message": "Feedback submitted successfully! Thank you."})
    except Exception as e:
        conn.rollback()
        return jsonify({"message": f"Error: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

# ============================================================
# CHATBOT API
# ============================================================
GROQ_SYSTEM_PROMPT = """You are "Case Guide AI" — an AI legal assistant built specifically for Indian citizens on the Case Guide India platform.

=== YOUR IDENTITY ===
Your name is Case Guide AI. You were created as part of a BSc Data Science final year project by Janvi R. Patil and Ankita S. Ratan at Shree L.R. Tiwari Degree College, Mira Road.

=== CONVERSATION RULES ===

RULE 1 — GREETINGS & CASUAL CHAT:
If the user says hello, hi, how are you, good morning, thanks, thank you, or any casual greeting:
→ Reply warmly and briefly in 1-2 sentences, then ask how you can help with their legal issue.

RULE 2 — INDIAN LEGAL QUESTIONS (your main job):
Answer ONLY questions related to Indian legal topics:
- FIR filing and police complaints
- Cybercrime — Helpline: 1930, Website: cybercrime.gov.in
- Domestic Violence (Section 498A IPC) — Helpline: 1091
- Consumer complaints (Consumer Protection Act 2019) — Website: edaakhil.nic.in
- RTI (Right to Information Act 2005) — Website: rtionline.gov.in
- Property disputes (RERA, Transfer of Property Act)
- Women Safety — Helpline: 1091, Emergency: 112
- Free Legal Aid (NALSA) — Helpline: 15100
- Child Safety (POCSO Act 2012) — Childline: 1098
- General Indian laws, IPC sections, court procedures

For legal questions: give step-by-step answers, mention relevant laws, helpline numbers, and format with headings and bullet points.

RULE 3 — OFF-TOPIC QUESTIONS:
If the user asks anything not related to Indian legal matters:
→ Politely say you are specialized only in Indian legal matters and redirect them.

RULE 4 — STAY IN CHARACTER:
Never say you are LLaMA or Groq. You are Case Guide AI.

=== FORMATTING ===
- Use **bold** for important terms
- Use bullet points for lists
- Use numbered lists for step-by-step procedures"""


@app.route("/chat", methods=["POST"])
def chat():
    data         = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type a message."}), 400

    session_id = session.get("user_id", "guest")

    if session_id not in conversation_store:
        conversation_store[session_id] = []
    history = conversation_store[session_id]

    try:
        messages = [{"role": "system", "content": GROQ_SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant",  "content": turn["bot"]})
        messages.append({"role": "user", "content": user_message})

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        reply = response.choices[0].message.content.strip()

        history.append({"user": user_message, "bot": reply})
        if len(history) > 10:
            history.pop(0)
        conversation_store[session_id] = history

    except Exception as e:
        print(f"[Groq error] {e}")
        reply = "Sorry, I encountered an error. Please try again."

    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO chat_messages (session_id, sender, message) VALUES (%s, %s, %s)",
            (str(session_id), "user", user_message)
        )
        cur.execute(
            "INSERT INTO chat_messages (session_id, sender, message) VALUES (%s, %s, %s)",
            (str(session_id), "bot", reply)
        )
        conn.commit()
    except Exception as e:
        print(f"Chat save skipped: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return jsonify({"reply": reply})


@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    session_id = session.get("user_id", "guest")
    if session_id in conversation_store:
        del conversation_store[session_id]
    return jsonify({"status": "cleared"})


@app.route("/chat-test")
def chat_test():
    return jsonify({"status": "Groq Chatbot is running!", "model": "llama-3.3-70b-versatile"})

# ============================================================
# ML CLASSIFY API
# ============================================================
@app.route("/classify", methods=["POST"])
def classify_query():
    data  = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"message": "Query is required"}), 400

    try:
        if ml_model and ml_vectorizer:
            vec        = ml_vectorizer.transform([query.lower()])
            category   = ml_model.predict(vec)[0]
            confidence = round(ml_model.predict_proba(vec).max() * 100)
        else:
            category   = "General"
            confidence = 0
    except Exception:
        category   = "General"
        confidence = 0

    next_steps = {
        "FIR":               "Visit your nearest police station to file an FIR.",
        "Cybercrime":        "Call cybercrime helpline 1930 or visit cybercrime.gov.in.",
        "Domestic Violence": "Call women helpline 1091 for immediate help.",
        "Consumer":          "File complaint at edaakhil.nic.in or District Consumer Forum.",
        "RTI":               "File RTI at rtionline.gov.in with Rs. 10 fee.",
        "Property":          "Consult a civil lawyer or file at civil court.",
        "Women Safety":      "Call 112 for emergency or 1091 for women helpline.",
        "Legal Aid":         "Call NALSA at 15100 for free legal help.",
        "Child Safety":      "Call Childline 1098 - available 24/7.",
        "General":           "Visit our Guidance page or call NALSA helpline at 15100.",
    }

    return jsonify({
        "query":      query,
        "category":   category,
        "confidence": confidence,
        "message":    next_steps.get(category, next_steps["General"])
    })

# ============================================================
# POLICE CONTACTS API
# ============================================================
@app.route("/police_contacts", methods=["GET"])
def get_police_contacts():
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT name, address, phone FROM police_stations LIMIT 5")
        rows = cur.fetchall()
        result = []
        for row in rows:
            result.append({
                "name":    row[0],
                "address": row[1],
                "phone":   row[2]
            })
        return jsonify(result)
    except Exception:
        return jsonify([])
    finally:
        cur.close()
        conn.close()

# ============================================================
# DASHBOARD DATA API
# ============================================================
@app.route("/user_dashboard", methods=["GET"])
def user_dashboard():
    username = session.get("username", None)
    if not username:
        return jsonify({"error": "not_logged_in"}), 401

    name_filter = request.args.get("name", "").strip()

    conn = get_db()
    cur  = conn.cursor()
    try:
        if name_filter:
            cur.execute("SELECT COUNT(*) FROM queries WHERE LOWER(user_name) = LOWER(%s) OR LOWER(user_name) = LOWER(%s)", (username, name_filter))
        else:
            # FIX: filter by logged-in user only, not all users
            cur.execute("SELECT COUNT(*) FROM queries WHERE LOWER(user_name) = LOWER(%s)", (username,))
        total_queries = cur.fetchone()[0]

        if name_filter:
            cur.execute("""
                SELECT category, COUNT(*) as cnt FROM queries
                WHERE LOWER(user_name) = LOWER(%s) OR LOWER(user_name) = LOWER(%s)
                GROUP BY category ORDER BY cnt DESC
            """, (username, name_filter))
        else:
            cur.execute("""
                SELECT category, COUNT(*) as cnt FROM queries
                WHERE LOWER(user_name) = LOWER(%s)
                GROUP BY category ORDER BY cnt DESC
            """, (username,))
        category_rows = cur.fetchall()
        categories = {row[0]: row[1] for row in category_rows}
        top_category = category_rows[0][0] if category_rows else "None"

        if name_filter:
            cur.execute("""
                SELECT query, category, created_at, user_name FROM queries
                WHERE LOWER(user_name) = LOWER(%s) OR LOWER(user_name) = LOWER(%s)
                ORDER BY id DESC LIMIT 20
            """, (username, name_filter))
        else:
            cur.execute("""
                SELECT query, category, created_at, user_name FROM queries
                WHERE LOWER(user_name) = LOWER(%s)
                ORDER BY id DESC LIMIT 20
            """, (username,))
        recent = []
        for row in cur.fetchall():
            recent.append({
                "query":      row[0],
                "category":   row[1] or "General",
                "created_at": str(row[2]),
                "user_name":  row[3]
            })

        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE sender = 'user'")
        total_chats = cur.fetchone()[0]

        if name_filter:
            cur.execute("""
                SELECT COUNT(DISTINCT DATE(created_at)) FROM queries
                WHERE LOWER(user_name) = LOWER(%s) OR LOWER(user_name) = LOWER(%s)
            """, (username, name_filter))
        else:
            cur.execute("""
                SELECT COUNT(DISTINCT DATE(created_at)) FROM queries
                WHERE LOWER(user_name) = LOWER(%s)
            """, (username,))
        days_active = cur.fetchone()[0]

        return jsonify({
            "username":      username,
            "total_queries": total_queries,
            "total_chats":   total_chats,
            "days_active":   days_active,
            "top_category":  top_category,
            "categories":    categories,
            "recent":        recent
        })
    except Exception as e:
        print(f"user_dashboard error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM queries")
        total_queries = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM user_feedback")
        total_feedback = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM chat_messages WHERE sender = 'user'")
        total_chats = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM case_guidance")
        total_guidance = cur.fetchone()[0]

        cur.execute("""
            SELECT user_name, query, category, status, created_at
            FROM queries ORDER BY id DESC LIMIT 10
        """)
        rows = cur.fetchall()
        recent_queries = []
        for row in rows:
            recent_queries.append({
                "user_name":  row[0] or "Anonymous",
                "query":      row[1],
                "category":   row[2] or "General",
                "status":     row[3] or "pending",
                "created_at": str(row[4])
            })

        return jsonify({
            "queries":        total_queries,
            "feedback":       total_feedback,
            "chats":          total_chats,
            "guidance":       total_guidance,
            "recent_queries": recent_queries
        })

    except Exception as e:
        print(f"Dashboard error: {e}")
        return jsonify({
            "queries": 0, "feedback": 0,
            "chats": 0,   "guidance": 0,
            "recent_queries": []
        })
    finally:
        cur.close()
        conn.close()

# ============================================================
# FEEDBACK SENTIMENT API — FIXED: now returns avg_rating + total
# ============================================================
@app.route("/feedback_sentiment", methods=["GET"])
def feedback_sentiment():
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT rating, comments FROM user_feedback")
        rows = cur.fetchall()

        positive    = 0
        neutral     = 0
        negative    = 0
        star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total_rating = 0

        for row in rows:
            rating = row[0]
            if rating in star_counts:
                star_counts[rating] += 1
                total_rating += rating
            if rating >= 4:
                positive += 1
            elif rating == 3:
                neutral += 1
            else:
                negative += 1

        total = len(rows)
        avg_rating = round(total_rating / total, 1) if total > 0 else 0

        return jsonify({
            "positive":    positive,
            "neutral":     neutral,
            "negative":    negative,
            "star_counts": star_counts,
            "avg_rating":  avg_rating,   # FIX: was missing before
            "total":       total          # FIX: was missing before
        })
    except Exception as e:
        return jsonify({"positive": 0, "neutral": 0, "negative": 0, "star_counts": {}, "avg_rating": 0, "total": 0})
    finally:
        cur.close()
        conn.close()

# ============================================================
# QUERY TRENDS API
# ============================================================
@app.route("/query_trends", methods=["GET"])
def query_trends():
    conn = get_db()
    cur  = conn.cursor()
    try:
        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM queries GROUP BY category ORDER BY count DESC
        """)
        rows = cur.fetchall()
        category_trends = [{"category": r[0] or "General", "count": r[1]} for r in rows]

        cur.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM queries GROUP BY day ORDER BY day DESC LIMIT 7
        """)
        rows = cur.fetchall()
        daily_trends = [{"date": str(r[0]), "count": r[1]} for r in rows]

        return jsonify({
            "category_trends": category_trends,
            "daily_trends":    daily_trends
        })
    except Exception as e:
        return jsonify({"category_trends": [], "daily_trends": []})
    finally:
        cur.close()
        conn.close()

# ============================================================
# LIVE STATS API
# ============================================================
@app.route("/live_stats", methods=["GET"])
def live_stats():
    conn = get_db()
    cur  = conn.cursor()
    try:
        try:
            cur.execute("""
                SELECT category,
                       COUNT(*)                        AS total,
                       ROUND(AVG(confidence)::numeric, 1) AS avg_conf,
                       SUM(CASE WHEN status='correct' THEN 1 ELSE 0 END) AS correct_count
                FROM queries
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category
                ORDER BY total DESC
            """)
        except Exception:
            conn.rollback()
            cur.execute("""
                SELECT category,
                       COUNT(*) AS total,
                       NULL     AS avg_conf,
                       NULL     AS correct_count
                FROM queries
                WHERE category IS NOT NULL AND category != ''
                GROUP BY category
                ORDER BY total DESC
            """)

        rows = cur.fetchall()
        total_all = sum(r[1] for r in rows) or 1

        per_category = {}
        for row in rows:
            cat         = row[0]
            count       = row[1]
            avg_conf    = float(row[2]) if row[2] is not None else None
            correct     = row[3] if row[3] is not None else None
            pct_of_all  = round((count / total_all) * 100, 1)
            correct_rate = round((correct / count) * 100, 1) if correct is not None and count > 0 else None

            per_category[cat] = {
                "count":        count,
                "pct_of_all":   pct_of_all,
                "avg_conf":     avg_conf,
                "correct_rate": correct_rate
            }

        return jsonify({
            "total_queries": total_all,
            "per_category":  per_category
        })

    except Exception as e:
        print(f"live_stats error: {e}")
        return jsonify({"total_queries": 0, "per_category": {}})
    finally:
        cur.close()
        conn.close()

# ============================================================
# MODEL INFO API
# ============================================================
@app.route("/model_info", methods=["GET"])
def model_info():
    import json
    try:
        with open("ml_model/model_info.json", "r") as f:
            info = json.load(f)
        return jsonify(info)
    except FileNotFoundError:
        return jsonify({
            "best_model":    "Linear SVM",
            "accuracy":      70.8,
            "all_models":    {
                "Logistic Regression": 68.8,
                "Naive Bayes":         66.7,
                "Linear SVM":          70.8
            },
            "per_category": {
                "Child Safety":      85.7,
                "RTI":               85.7,
                "Consumer":          72.7,
                "Domestic Violence": 76.9,
                "FIR":               66.7,
                "Legal Aid":         66.7,
                "Property":          66.7,
                "Cybercrime":        60.0,
                "Women Safety":      57.1,
                "General":           50.0
            },
            "total_samples": 240,
            "train_samples": 192,
            "test_samples":  48,
            "categories":    ["Child Safety","Consumer","Cybercrime","Domestic Violence",
                               "FIR","General","Legal Aid","Property","RTI","Women Safety"],
            "tfidf_features": 5000,
            "ngram_range":   "(1, 2)",
            "note":          "Run train_model.py to generate live data"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))