from flask import Flask, render_template, request
import os
import joblib
import fitz
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
reg = joblib.load('model/reg.pkl')
tfidf = joblib.load('model/tfidf.pkl')

# Skills for matching
skill_keywords = [
    'python', 'django', 'flask', 'html', 'css', 'javascript', 'react',
    'sql', 'git', 'docker', 'kubernetes', 'java', 'tensorflow', 'pandas',
    'communication', 'teamwork', 'problem-solving', 'aws', 'azure', 'api'
]

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return " ".join(page.get_text() for page in doc).strip()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def extract_skills(text):
    return [skill for skill in skill_keywords if skill in text.lower()]

# ✅ New: Resume validation logic
def is_resume(text):
    resume_indicators = [
        "education", "experience", "skills", "projects", "summary",
        "certifications", "contact", "linkedin", "github", "bachelor", "master", "email"
    ]
    text = text.lower()
    score = sum(1 for word in resume_indicators if word in text)
    return len(text) >= 500 and score >= 3

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['resume']
    jd_text = request.form['jd']

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)

        # ✅ Resume check
        if not is_resume(resume_text):
            return render_template('result.html', is_invalid=True)

        combined_text = clean_text(resume_text + " " + jd_text)
        X_input = tfidf.transform([combined_text])
        score = round(reg.predict(X_input)[0], 2)

        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        matched_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))

        match_status = 'Matched' if score >= 70 else 'Not Matched'

        return render_template(
            'result.html',
            match_status=match_status,
            compatibility_score=score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            is_invalid=False
        )
    else:
        return "❌ Only PDF files are supported."

if __name__ == '__main__':
    app.run(debug=True)
