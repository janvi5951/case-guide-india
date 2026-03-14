import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ============================================================
# ML CLASSIFIER — Case Guide India
# Run this file ONCE to train and save the model
# app.py will load it automatically after that
# ============================================================

training_data = [

    # ══ FIR / Police (30 examples) ══
    ("how to file fir",                            "FIR"),
    ("file a police complaint",                    "FIR"),
    ("police not registering my complaint",        "FIR"),
    ("how to report a crime",                      "FIR"),
    ("stolen wallet what to do",                   "FIR"),
    ("my phone was stolen",                        "FIR"),
    ("someone robbed me",                          "FIR"),
    ("lodge complaint at police station",          "FIR"),
    ("theft complaint police",                     "FIR"),
    ("police refusing to file fir",                "FIR"),
    ("robbery at home",                            "FIR"),
    ("assault complaint police",                   "FIR"),
    ("how to check fir status",                    "FIR"),
    ("lost documents complaint",                   "FIR"),
    ("kidnapping complaint police",                "FIR"),
    ("bike stolen complaint",                      "FIR"),
    ("car theft police report",                    "FIR"),
    ("cash stolen from house",                     "FIR"),
    ("neighbour beating me",                       "FIR"),
    ("fight in street complaint",                  "FIR"),
    ("murder attempt complaint",                   "FIR"),
    ("threatening calls complaint police",         "FIR"),
    ("missing person police complaint",            "FIR"),
    ("first information report process",           "FIR"),
    ("can i file fir online",                      "FIR"),
    ("copy of fir how to get",                     "FIR"),
    ("fir against unknown person",                 "FIR"),
    ("zero fir what is it",                        "FIR"),
    ("police harassment complaint",                "FIR"),
    ("false case filed against me",                "FIR"),

    # ══ Cybercrime (30 examples) ══
    ("online fraud happened to me",                "Cybercrime"),
    ("how to report cybercrime",                   "Cybercrime"),
    ("money deducted from account by fraud",       "Cybercrime"),
    ("someone hacked my account",                  "Cybercrime"),
    ("fake website took my money",                 "Cybercrime"),
    ("i received a fraud call",                    "Cybercrime"),
    ("identity theft online",                      "Cybercrime"),
    ("cyber fraud complaint",                      "Cybercrime"),
    ("someone is blackmailing me online",          "Cybercrime"),
    ("cyberbullying complaint",                    "Cybercrime"),
    ("otp fraud bank",                             "Cybercrime"),
    ("fake profile made on my name",               "Cybercrime"),
    ("phishing email scam",                        "Cybercrime"),
    ("how to report online scam",                  "Cybercrime"),
    ("social media harassment",                    "Cybercrime"),
    ("upi fraud complaint",                        "Cybercrime"),
    ("whatsapp fraud money lost",                  "Cybercrime"),
    ("email hacked what to do",                    "Cybercrime"),
    ("online shopping fraud refund",               "Cybercrime"),
    ("credit card fraud online",                   "Cybercrime"),
    ("fake job offer scam",                        "Cybercrime"),
    ("cyber cell complaint how to file",           "Cybercrime"),
    ("morphed photos being circulated",            "Cybercrime"),
    ("ransom demand online",                       "Cybercrime"),
    ("data breach personal information",           "Cybercrime"),
    ("dating app fraud money",                     "Cybercrime"),
    ("investment fraud online",                    "Cybercrime"),
    ("cybercrime helpline number",                 "Cybercrime"),
    ("phone stolen and misused",                   "Cybercrime"),
    ("bank account hacked online",                 "Cybercrime"),

    # ══ Domestic Violence (30 examples) ══
    ("husband is beating me",                      "Domestic Violence"),
    ("domestic violence help",                     "Domestic Violence"),
    ("wife is being abused at home",               "Domestic Violence"),
    ("dowry harassment complaint",                 "Domestic Violence"),
    ("in laws harassing me",                       "Domestic Violence"),
    ("498a complaint how to file",                 "Domestic Violence"),
    ("physical abuse by husband",                  "Domestic Violence"),
    ("mental torture by family",                   "Domestic Violence"),
    ("how to get protection order",                "Domestic Violence"),
    ("domestic abuse helpline",                    "Domestic Violence"),
    ("husband threatening to kill me",             "Domestic Violence"),
    ("cruelty by husband ipc",                     "Domestic Violence"),
    ("how to leave abusive relationship legally",  "Domestic Violence"),
    ("domestic violence act complaint",            "Domestic Violence"),
    ("husband not giving maintenance",             "Domestic Violence"),
    ("thrown out of house by husband",             "Domestic Violence"),
    ("mother in law harassing",                    "Domestic Violence"),
    ("dowry demand by in laws",                    "Domestic Violence"),
    ("emotional abuse by spouse",                  "Domestic Violence"),
    ("husband drinks and beats",                   "Domestic Violence"),
    ("domestic violence shelter home",             "Domestic Violence"),
    ("protection officer domestic violence",       "Domestic Violence"),
    ("restraining order against husband",          "Domestic Violence"),
    ("husband took my jewellery",                  "Domestic Violence"),
    ("financial abuse by husband",                 "Domestic Violence"),
    ("forced to leave matrimonial home",           "Domestic Violence"),
    ("domestic violence helpline number",          "Domestic Violence"),
    ("husband not allowing to work",               "Domestic Violence"),
    ("beaten by family members",                   "Domestic Violence"),
    ("domestic abuse evidence",                    "Domestic Violence"),

    # ══ Consumer (30 examples) ══
    ("defective product complaint",                "Consumer"),
    ("company not giving refund",                  "Consumer"),
    ("how to file consumer complaint",             "Consumer"),
    ("bought fake product online",                 "Consumer"),
    ("cheated by company",                         "Consumer"),
    ("poor service complaint",                     "Consumer"),
    ("bank charged extra fees",                    "Consumer"),
    ("insurance company not paying claim",         "Consumer"),
    ("hospital overcharged me",                    "Consumer"),
    ("ecommerce fraud complaint",                  "Consumer"),
    ("product not delivered refund",               "Consumer"),
    ("how to approach consumer court",             "Consumer"),
    ("electricity bill complaint",                 "Consumer"),
    ("telecom company fraud",                      "Consumer"),
    ("amazon flipkart fraud complaint",            "Consumer"),
    ("mobile phone repair fraud",                  "Consumer"),
    ("gym membership refund",                      "Consumer"),
    ("airline ticket refund not given",            "Consumer"),
    ("hotel service poor complaint",               "Consumer"),
    ("medicine expired sold to me",                "Consumer"),
    ("car service centre fraud",                   "Consumer"),
    ("builder not returning money",                "Consumer"),
    ("edaakhil complaint how to file",             "Consumer"),
    ("wrong billing complaint",                    "Consumer"),
    ("food delivery wrong order refund",           "Consumer"),
    ("internet service provider complaint",        "Consumer"),
    ("consumer protection act rights",             "Consumer"),
    ("district consumer forum complaint",          "Consumer"),
    ("unfair trade practice complaint",            "Consumer"),
    ("warranty not honoured complaint",            "Consumer"),

    # ══ RTI (20 examples) ══
    ("how to file rti",                            "RTI"),
    ("right to information application",           "RTI"),
    ("how to get government information",          "RTI"),
    ("rti application process",                    "RTI"),
    ("government not giving information",          "RTI"),
    ("rti fee how much",                           "RTI"),
    ("how to file rti online",                     "RTI"),
    ("rti reply not received",                     "RTI"),
    ("first appeal rti process",                   "RTI"),
    ("public information officer contact",         "RTI"),
    ("rti for government scheme information",      "RTI"),
    ("rti against police",                         "RTI"),
    ("rti for court case information",             "RTI"),
    ("how long does rti take",                     "RTI"),
    ("rti rejected what to do",                    "RTI"),
    ("second appeal rti",                          "RTI"),
    ("central information commission",             "RTI"),
    ("rti for bpl card information",               "RTI"),
    ("rti for land records",                       "RTI"),
    ("rti application format",                     "RTI"),

    # ══ Property (20 examples) ══
    ("property dispute with neighbour",            "Property"),
    ("land grabbing complaint",                    "Property"),
    ("tenant not paying rent",                     "Property"),
    ("illegal possession of my land",              "Property"),
    ("builder fraud complaint",                    "Property"),
    ("rera complaint how to file",                 "Property"),
    ("inheritance property dispute",               "Property"),
    ("how to evict tenant",                        "Property"),
    ("property registration issue",                "Property"),
    ("encroachment on my land",                    "Property"),
    ("boundary dispute with neighbour",            "Property"),
    ("flat possession delayed by builder",         "Property"),
    ("property documents forged",                  "Property"),
    ("ancestral property dispute",                 "Property"),
    ("rent agreement dispute",                     "Property"),
    ("property tax complaint",                     "Property"),
    ("illegal construction complaint",             "Property"),
    ("land measurement dispute",                   "Property"),
    ("property sale fraud",                        "Property"),
    ("tenancy rights in india",                    "Property"),

    # ══ Women Safety (20 examples) ══
    ("being stalked by someone",                   "Women Safety"),
    ("eve teasing complaint",                      "Women Safety"),
    ("workplace harassment complaint",             "Women Safety"),
    ("sexual harassment at office",                "Women Safety"),
    ("women safety helpline",                      "Women Safety"),
    ("complaint against stalker",                  "Women Safety"),
    ("molested in public place",                   "Women Safety"),
    ("online harassment of women",                 "Women Safety"),
    ("posh act complaint",                         "Women Safety"),
    ("internal complaints committee",              "Women Safety"),
    ("rape complaint how to file",                 "Women Safety"),
    ("outrage of modesty complaint",               "Women Safety"),
    ("groping in bus complaint",                   "Women Safety"),
    ("obscene messages from unknown",              "Women Safety"),
    ("women helpline 1091",                        "Women Safety"),
    ("acid attack complaint",                      "Women Safety"),
    ("forced marriage complaint",                  "Women Safety"),
    ("honour crime complaint",                     "Women Safety"),
    ("women police station complaint",             "Women Safety"),
    ("sexual assault complaint",                   "Women Safety"),

    # ══ Legal Aid (20 examples) ══
    ("need free lawyer",                           "Legal Aid"),
    ("how to get free legal help",                 "Legal Aid"),
    ("legal aid services india",                   "Legal Aid"),
    ("dlsa legal aid application",                 "Legal Aid"),
    ("cannot afford lawyer",                       "Legal Aid"),
    ("free legal advice india",                    "Legal Aid"),
    ("nalsa helpline 15100",                       "Legal Aid"),
    ("lok adalat how to apply",                    "Legal Aid"),
    ("legal services authority contact",           "Legal Aid"),
    ("pro bono lawyer india",                      "Legal Aid"),
    ("legal help for poor people",                 "Legal Aid"),
    ("free lawyer for women india",                "Legal Aid"),
    ("district legal services",                    "Legal Aid"),
    ("legal aid for sc st",                        "Legal Aid"),
    ("mediation centre india",                     "Legal Aid"),
    ("legal aid eligibility india",                "Legal Aid"),
    ("court fee waiver poor",                      "Legal Aid"),
    ("free legal camp",                            "Legal Aid"),
    ("paralegal services india",                   "Legal Aid"),
    ("legal advice helpline",                      "Legal Aid"),

    # ══ Child Safety (20 examples) ══
    ("child abuse complaint",                      "Child Safety"),
    ("pocso act complaint",                        "Child Safety"),
    ("childline helpline 1098",                    "Child Safety"),
    ("child sexual abuse report",                  "Child Safety"),
    ("missing child complaint",                    "Child Safety"),
    ("child labour complaint",                     "Child Safety"),
    ("child being harassed at school",             "Child Safety"),
    ("how to report child abuse",                  "Child Safety"),
    ("child trafficking complaint",                "Child Safety"),
    ("juvenile justice complaint",                 "Child Safety"),
    ("child marriage complaint",                   "Child Safety"),
    ("school teacher abusing child",               "Child Safety"),
    ("child begging complaint",                    "Child Safety"),
    ("child rights violation",                     "Child Safety"),
    ("adoption fraud complaint",                   "Child Safety"),
    ("child custody dispute",                      "Child Safety"),
    ("online predator targeting child",            "Child Safety"),
    ("child pornography report",                   "Child Safety"),
    ("child neglect complaint",                    "Child Safety"),
    ("corporal punishment school",                 "Child Safety"),

    # ══ General (20 examples) ══
    ("what are my legal rights",                   "General"),
    ("how does indian law work",                   "General"),
    ("legal advice needed",                        "General"),
    ("what is bail process",                       "General"),
    ("what is legal notice",                       "General"),
    ("how to find a good lawyer",                  "General"),
    ("civil vs criminal case difference",          "General"),
    ("what is affidavit",                          "General"),
    ("how to file case in court",                  "General"),
    ("legal help emergency india",                 "General"),
    ("what is anticipatory bail",                  "General"),
    ("power of attorney process",                  "General"),
    ("what is cognizable offence",                 "General"),
    ("legal heir certificate",                     "General"),
    ("succession certificate process",             "General"),
    ("notary affidavit process",                   "General"),
    ("court summons received",                     "General"),
    ("legal rights of accused",                    "General"),
    ("mediation vs court",                         "General"),
    ("what is section 420",                        "General"),
]

# ── Convert to DataFrame ──────────────────────────────────────
df = pd.DataFrame(training_data, columns=["query", "category"])
X  = df["query"]
y  = df["category"]

print(f"✅ Total training samples: {len(df)}")
print(f"📊 Category distribution:")
for cat, count in df['category'].value_counts().items():
    print(f"   {cat}: {count}")
print()

# ── Split 80/20 ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"🔀 Training: {len(X_train)} | Testing: {len(X_test)}")
print()

# ── TF-IDF Vectorizer ─────────────────────────────────────────
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=1)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# ── Train Model ───────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=5, random_state=42)
model.fit(X_train_vec, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred   = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.1f}%")
print()
print("📋 Detailed Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── Save Model ────────────────────────────────────────────────
os.makedirs("ml_model", exist_ok=True)
with open("ml_model/classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("ml_model/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("💾 Saved → ml_model/classifier.pkl")
print("💾 Saved → ml_model/vectorizer.pkl")
print()

# ── Quick Test ────────────────────────────────────────────────
print("🧪 Quick Test — let's see if it classifies correctly:")
test_queries = [
    "my phone was stolen",
    "husband is beating me",
    "online fraud money lost",
    "builder not giving flat",
    "need free lawyer",
    "child being abused",
]
for q in test_queries:
    vec  = tfidf.transform([q])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    print(f"   '{q}' → {pred} ({prob*100:.0f}% confident)")

print()
print("✅ ML Model is ready! Now copy ml_model/ folder to your project.")