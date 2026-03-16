-- ============================================================
-- Case Guide India — Database Setup for Render PostgreSQL
-- ============================================================
-- Run this SQL ONCE in your Render PostgreSQL database
-- How to run: Render Dashboard → your PostgreSQL → PSQL button → paste this
-- ============================================================

-- 1. USERS TABLE
CREATE TABLE IF NOT EXISTS app_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(100) NOT NULL,
    email         VARCHAR(150) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. QUERIES TABLE
CREATE TABLE IF NOT EXISTS queries (
    id         SERIAL PRIMARY KEY,
    user_name  VARCHAR(100),
    query      TEXT NOT NULL,
    category   VARCHAR(100),
    confidence INTEGER DEFAULT 0,
    status     VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. CHAT MESSAGES TABLE
CREATE TABLE IF NOT EXISTS chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    sender     VARCHAR(20),
    message    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. FEEDBACK TABLE
CREATE TABLE IF NOT EXISTS user_feedback (
    id         SERIAL PRIMARY KEY,
    rating     INTEGER CHECK (rating BETWEEN 1 AND 5),
    comments   TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. LEGAL GUIDANCE TABLE
CREATE TABLE IF NOT EXISTS case_guidance (
    id          SERIAL PRIMARY KEY,
    case_type   VARCHAR(100) NOT NULL,
    description TEXT,
    steps       TEXT,
    authority   VARCHAR(200),
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. POLICE STATIONS TABLE
CREATE TABLE IF NOT EXISTS police_stations (
    id      SERIAL PRIMARY KEY,
    name    VARCHAR(200),
    address TEXT,
    phone   VARCHAR(50)
);

-- ============================================================
-- SEED DATA — Legal Guidance (10 categories)
-- ============================================================
INSERT INTO case_guidance (case_type, description, steps, authority) VALUES
('FIR', 'Filing a First Information Report with the police for cognizable offences.',
 '1. Visit the nearest police station\n2. Explain your complaint to the duty officer\n3. Ask them to write and register the FIR\n4. Get a copy of the FIR with FIR number\n5. If police refuse, approach the Superintendent of Police\n6. You can also file online FIR at your state police portal',
 'Nearest Police Station / Superintendent of Police'),

('Cybercrime', 'Reporting online fraud, hacking, phishing, or cyber harassment.',
 '1. Call Cybercrime Helpline 1930 immediately\n2. Visit cybercrime.gov.in to file online complaint\n3. Preserve all digital evidence — screenshots, emails, messages\n4. Inform your bank immediately if money was deducted\n5. Block your debit/credit card via net banking\n6. File FIR at your nearest Cyber Crime Police Station',
 'National Cyber Crime Reporting Portal — cybercrime.gov.in / Helpline: 1930'),

('Domestic Violence', 'Protection under the Domestic Violence Act 2005 and IPC Section 498A.',
 '1. Call Women Helpline 1091 for immediate help\n2. Visit the nearest women police station\n3. Contact a Protection Officer under DV Act 2005\n4. File a Domestic Incident Report (DIR)\n5. Apply for Protection Order from Magistrate\n6. You can also approach an NGO or legal aid center for support',
 'Women Police Station / Protection Officer / Mahila Court / Helpline: 1091'),

('Consumer', 'Filing complaints under the Consumer Protection Act 2019 for defective products or services.',
 '1. Keep all purchase receipts, bills, and warranty cards\n2. Send a written complaint to the company/seller first\n3. If unresolved in 30 days, file at edaakhil.nic.in\n4. For claims below Rs. 50 lakh go to District Consumer Forum\n5. For Rs. 50 lakh–2 crore go to State Consumer Forum\n6. No lawyer needed for cases below Rs. 1 crore',
 'District Consumer Disputes Redressal Commission / edaakhil.nic.in'),

('RTI', 'Filing a Right to Information application under the RTI Act 2005.',
 '1. Identify the Public Authority holding the information\n2. Write a simple application addressing the Public Information Officer (PIO)\n3. Pay application fee of Rs. 10 (BPL applicants exempted)\n4. Submit at rtionline.gov.in or by post/in person\n5. PIO must respond within 30 days\n6. If no response, file First Appeal with First Appellate Authority',
 'Public Information Officer (PIO) of the relevant Department / rtionline.gov.in'),

('Property', 'Resolving property disputes, illegal possession, or landlord-tenant issues.',
 '1. Collect all property documents — sale deed, title deed, tax receipts\n2. Send a legal notice to the other party first\n3. File complaint at local police station for illegal possession (Section 145 CrPC)\n4. For RERA registered properties, file at your State RERA Authority\n5. File civil suit in District Civil Court for ownership disputes\n6. Consult a civil lawyer for title verification',
 'District Civil Court / State RERA Authority / Local Police Station'),

('Women Safety', 'Protection from stalking, harassment, assault and other crimes against women.',
 '1. Call 112 for emergency or 1091 for Women Helpline\n2. File FIR at the nearest police station immediately\n3. Stalking is a crime under Section 354D IPC — insist police register FIR\n4. Workplace harassment — file complaint with your company Internal Complaints Committee (ICC) under POSH Act\n5. Take medical examination if physically assaulted\n6. Save all evidence — messages, photos, call logs',
 'Nearest Police Station / Women Helpline: 1091 / Emergency: 112'),

('Legal Aid', 'Getting free legal assistance through the National Legal Services Authority (NALSA).',
 '1. Call NALSA Helpline 15100 for free legal guidance\n2. Visit your District Legal Services Authority (DLSA) office\n3. Eligible: BPL families, women, SC/ST, disabled persons, prisoners, children\n4. Fill the legal aid application form at DLSA\n5. A free lawyer will be assigned to your case\n6. You can also approach your State Legal Services Authority (SLSA)',
 'District Legal Services Authority (DLSA) / NALSA Helpline: 15100'),

('Child Safety', 'Protection of children from abuse, neglect, and exploitation under POCSO Act 2012.',
 '1. Call Childline 1098 — free, 24/7, available across India\n2. Report child abuse to the nearest police station\n3. Child abuse is covered under POCSO Act 2012 — police must register FIR\n4. Contact Child Welfare Committee (CWC) in your district\n5. Approach National Commission for Protection of Child Rights (NCPCR)\n6. Keep the child safe and do not delay reporting',
 'Childline: 1098 / Child Welfare Committee / NCPCR'),

('General', 'General legal information and guidance for common legal questions.',
 '1. Identify what type of legal issue you are facing\n2. Contact NALSA helpline 15100 for free initial guidance\n3. Visit your nearest District Legal Services Authority (DLSA)\n4. For court matters, hire a lawyer registered with the Bar Council\n5. Lok Adalat is a free alternative dispute resolution option\n6. National Consumer Helpline: 1915',
 'District Legal Services Authority (DLSA) / NALSA: 15100');

-- ============================================================
-- SEED DATA — Police Stations (Mira Bhayandar area)
-- ============================================================
INSERT INTO police_stations (name, address, phone) VALUES
('Mira Road Police Station', 'Near Mira Road Railway Station, Mira Road East, Thane - 401107', '022-28555555'),
('Bhayandar Police Station', 'Bhayandar East, Thane District - 401105', '022-28195555'),
('Kashimira Police Station', 'Kashimira, Mira Road, Thane - 401104', '022-28457777'),
('Naigaon Police Station', 'Naigaon East, Vasai-Virar, Palghar - 401208', '0250-2510100'),
('Vasai Police Station', 'Vasai West, Palghar District - 401202', '0250-2340100');

-- ============================================================
-- Done! All tables created and seeded.
-- ============================================================
