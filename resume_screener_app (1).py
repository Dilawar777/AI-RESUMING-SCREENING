import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    body, .stApp, .main, .block-container {
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
    }
    * { color: #1A1A2E !important; }
    .main-title { font-size:2.2rem; font-weight:700; color:#1A56A0 !important; }
    .sub-title  { font-size:1rem; color:#6B7280 !important; }
    .kpi-card   { background:#F0F7FF; border-left:4px solid #1A56A0;
                  padding:14px 18px; border-radius:8px; margin:4px 0; }
    .kpi-val    { font-size:1.8rem; font-weight:700; color:#1A56A0 !important; }
    .kpi-lbl    { font-size:0.82rem; color:#6B7280 !important; }
    .strong     { background:#D1FAE5; border-left:4px solid #059669;
                  padding:12px 16px; border-radius:8px; color:#1A1A2E !important; margin:4px 0; }
    .moderate   { background:#FFF8E7; border-left:4px solid #F39C12;
                  padding:12px 16px; border-radius:8px; color:#1A1A2E !important; margin:4px 0; }
    .weak       { background:#FEE2E2; border-left:4px solid #DC2626;
                  padding:12px 16px; border-radius:8px; color:#1A1A2E !important; margin:4px 0; }
    .section    { font-size:1.1rem; font-weight:600; color:#1A1A2E !important;
                  border-bottom:2px solid #1A56A0;
                  padding-bottom:4px; margin:20px 0 12px; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

SKILL_KEYWORDS = {
    "Programming":      ["python", "sql", "java", "c++", "r", "javascript", "scala"],
    "Data Analysis":    ["pandas", "numpy", "excel", "eda", "statistical analysis",
                         "data analysis", "data cleaning", "data wrangling"],
    "Visualization":    ["tableau", "power bi", "matplotlib", "seaborn", "plotly", "looker"],
    "Machine Learning": ["scikit-learn", "machine learning", "random forest",
                         "logistic regression", "deep learning", "neural network",
                         "tensorflow", "pytorch", "xgboost", "svm", "k-means",
                         "clustering", "regression", "classification"],
    "Databases":        ["mysql", "postgresql", "mongodb", "sql server",
                         "database design", "jdbc", "nosql"],
    "Tools":            ["git", "github", "jupyter", "streamlit", "docker",
                         "aws", "azure", "gcp", "linux"],
    "Soft Skills":      ["communication", "teamwork", "leadership",
                         "problem solving", "analytical thinking", "collaboration"]
}

def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return re.sub(r'\s+', ' ', text).strip()

def extract_skills(text, skill_list):
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
            found.append(skill)
    return found

def extract_skills_by_category(text, skill_dict):
    return {cat: extract_skills(text, skills) for cat, skills in skill_dict.items()}

def calculate_tfidf_score(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)

st.markdown('<p class="main-title">📄 AI Resume Screener</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your resume · Paste any job description · Get an instant match score</p>', unsafe_allow_html=True)
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section">Upload Resume (PDF)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
with col2:
    st.markdown('<p class="section">Paste Job Description</p>', unsafe_allow_html=True)
    job_desc = st.text_area("", height=250,
                             placeholder="Paste the full job description here...",
                             label_visibility="collapsed")

analyze_btn = st.button("🔍 Analyse Resume", type="primary", use_container_width=True)

if analyze_btn:
    if not uploaded_file:
        st.error("Please upload your resume PDF first.")
    elif not job_desc.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analysing your resume..."):
            resume_text = extract_pdf_text(uploaded_file)
            resume_skills_by_cat = extract_skills_by_category(resume_text, SKILL_KEYWORDS)
            job_skills_by_cat    = extract_skills_by_category(job_desc,    SKILL_KEYWORDS)

            resume_skills_flat = [s for skills in resume_skills_by_cat.values() for s in skills]
            job_skills_flat    = [s for skills in job_skills_by_cat.values()    for s in skills]

            matched   = [s for s in job_skills_flat if s in resume_skills_flat]
            missing   = [s for s in job_skills_flat if s not in resume_skills_flat]
            match_pct = len(matched) / len(job_skills_flat) * 100 if job_skills_flat else 0

            tfidf_score  = calculate_tfidf_score(resume_text, job_desc)
            final_score  = match_pct * 0.7 + tfidf_score * 0.3

        st.markdown(""); st.divider()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{final_score:.0f}%</div><div class="kpi-lbl">Overall Match Score</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{match_pct:.0f}%</div><div class="kpi-lbl">Keyword Match</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(matched)}/{len(job_skills_flat)}</div><div class="kpi-lbl">Skills Matched</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(missing)}</div><div class="kpi-lbl">Skills Missing</div></div>', unsafe_allow_html=True)

        st.markdown("")

        if final_score >= 75:
            st.markdown(f'<div class="strong">✅ <b>STRONG MATCH ({final_score:.0f}%)</b> — Your resume is well-aligned with this job. Likely to pass ATS screening.</div>', unsafe_allow_html=True)
        elif final_score >= 50:
            st.markdown(f'<div class="moderate">⚠️ <b>MODERATE MATCH ({final_score:.0f}%)</b> — Good foundation but consider adding missing keywords.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="weak">❌ <b>WEAK MATCH ({final_score:.0f}%)</b> — Resume needs significant alignment with this job description.</div>', unsafe_allow_html=True)

        st.markdown("")
        tab1, tab2, tab3 = st.tabs(["✗ Missing Skills", "✓ Matched Skills", "📊 Skills by Category"])

        with tab1:
            st.markdown("<h4 style='color:#1A1A2E'>Keywords to Add to Your Resume</h4>", unsafe_allow_html=True)
            if missing:
                cols = st.columns(3)
                for i, skill in enumerate(missing):
                    cols[i % 3].markdown(f"<p style='color:#DC2626; font-weight:bold'>❌ {skill}</p>", unsafe_allow_html=True)
            else:
                st.success("Your resume contains all keywords from this job description!")

        with tab2:
            st.markdown("<h4 style='color:#1A1A2E'>Skills Your Resume Already Has</h4>", unsafe_allow_html=True)
            if matched:
                cols = st.columns(3)
                for i, skill in enumerate(matched):
                    cols[i % 3].markdown(f"<p style='color:#059669; font-weight:bold'>✅ {skill}</p>", unsafe_allow_html=True)
            else:
                st.warning("No matching skills found.")

        with tab3:
            st.markdown("<h4 style='color:#1A1A2E'>Resume Skills by Category</h4>", unsafe_allow_html=True)
            for category, skills in resume_skills_by_cat.items():
                if skills:
                    st.markdown(
                        f"<p style='color:#1A1A2E'><b style='color:#1A56A0'>{category}:</b> {', '.join(skills)}</p>",
                        unsafe_allow_html=True
                    )

st.divider()
st.markdown(
    "<p style='text-align:center;color:#9CA3AF;font-size:0.8rem;'>"
    "Built by <b>Dilawar Mahar</b> · Python · TF-IDF · pdfplumber · Streamlit · "
    "<a href='https://github.com/Dilawar777' style='color:#1A56A0'>GitHub</a></p>",
    unsafe_allow_html=True
)
