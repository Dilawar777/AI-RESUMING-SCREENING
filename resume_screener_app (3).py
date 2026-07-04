import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #FFFFFF; }
.stApp * { color: #1A1A2E; }
</style>
""", unsafe_allow_html=True)

SKILL_KEYWORDS = {
    "Programming":      ["python", "sql", "java", "c++", "r", "javascript"],
    "Data Analysis":    ["pandas", "numpy", "excel", "eda", "statistical analysis", "data cleaning"],
    "Visualization":    ["tableau", "power bi", "matplotlib", "seaborn", "plotly"],
    "Machine Learning": ["scikit-learn", "machine learning", "random forest", "logistic regression",
                         "tensorflow", "pytorch", "xgboost", "svm", "k-means", "regression", "classification"],
    "Databases":        ["mysql", "postgresql", "mongodb", "jdbc"],
    "Tools":            ["git", "github", "jupyter", "streamlit", "docker", "aws", "azure"],
    "Soft Skills":      ["communication", "teamwork", "leadership", "problem solving", "collaboration"]
}

def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return re.sub(r'\s+', ' ', text).strip()

def extract_skills(text, skill_list):
    found = []
    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower(), re.IGNORECASE):
            found.append(skill)
    return found

def tfidf_score(r, j):
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    m = v.fit_transform([r, j])
    return round(cosine_similarity(m[0:1], m[1:2])[0][0] * 100, 2)

st.title("📄 AI Resume Screener")
st.caption("Upload your resume · Paste job description · Get instant match score")
st.divider()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
with c2:
    st.subheader("Job Description")
    job_desc = st.text_area("Paste job description here", height=250)

if st.button("🔍 Analyse Resume", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload your resume PDF.")
    elif not job_desc.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analysing..."):
            resume_text = extract_pdf_text(uploaded_file)
            res_skills = {c: extract_skills(resume_text, s) for c, s in SKILL_KEYWORDS.items()}
            job_skills = {c: extract_skills(job_desc, s) for c, s in SKILL_KEYWORDS.items()}
            res_flat = [s for v in res_skills.values() for s in v]
            job_flat = [s for v in job_skills.values() for s in v]
            matched = [s for s in job_flat if s in res_flat]
            missing = [s for s in job_flat if s not in res_flat]
            kw_pct = len(matched) / len(job_flat) * 100 if job_flat else 0
            tf_score = tfidf_score(resume_text, job_desc)
            final = kw_pct * 0.7 + tf_score * 0.3

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Score", f"{final:.0f}%")
        k2.metric("Keyword Match", f"{kw_pct:.0f}%")
        k3.metric("Skills Matched", f"{len(matched)}/{len(job_flat)}")
        k4.metric("Skills Missing", len(missing))

        if final >= 75:
            st.success(f"✅ STRONG MATCH ({final:.0f}%) — Likely to pass ATS screening.")
        elif final >= 50:
            st.warning(f"⚠️ MODERATE MATCH ({final:.0f}%) — Add missing keywords to improve.")
        else:
            st.error(f"❌ WEAK MATCH ({final:.0f}%) — Resume needs significant alignment.")

        tab1, tab2, tab3 = st.tabs(["❌ Missing Skills", "✅ Matched Skills", "📊 Skills by Category"])

        with tab1:
            st.subheader("Keywords to Add to Your Resume")
            if missing:
                cols = st.columns(3)
                for i, s in enumerate(missing):
                    cols[i%3].write(f"❌ {s}")
            else:
                st.success("Your resume has all required keywords!")

        with tab2:
            st.subheader("Skills Your Resume Already Has")
            if matched:
                cols = st.columns(3)
                for i, s in enumerate(matched):
                    cols[i%3].write(f"✅ {s}")

        with tab3:
            st.subheader("Resume Skills by Category")
            for cat, skills in res_skills.items():
                if skills:
                    st.write(f"**{cat}:** {', '.join(skills)}")

st.divider()
st.caption("Built by Dilawar Mahar · Python · TF-IDF · pdfplumber · Streamlit")
