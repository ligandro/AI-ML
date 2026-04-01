# 🤖 AI Job Matcher - LangGraph Pipeline for Resume Matching & Cover Letters

An intelligent tool that analyzes a candidate's resume and a job description to evaluate compatibility and automatically generate a tailored cover letter. Ideal for 
personal use to check whether you fit a particular role and easily get a tailored cover letter.

---
![Demo](demo.gif)
## 🚀 Features

- 🔍 **Resume Parsing**: Extracts skills, experience, education, and achievements from the uploaded resume.
- 🧠 **Job Matching**: Compares parsed resume data against a job description to assess relevance.
- ✍️ **Cover Letter Generation**: Produces a personalized cover letter using AI based on the user's qualifications and the job role.
- 📊 **Skill & Domain Analysis**: Highlights key technical skills and domain expertise.
- 🧾 **Streamlit Interface**: Simple web UI to interactively upload resumes, input job descriptions, and get results instantly.

---
## 🛠️ Tech Stack


- **Ollama** : Advanced language model for natural language processing and AI-driven analysis.
- **Streamlit** : Modern web framework for building interactive user interfaces.
- **Swarm Framework** : Coordination of specialized AI agents for modular task execution.
- **Python** : Core programming language for backend logic and AI integration.

---
## 🧩 Architecture Overview

**Frontend** :

- Streamlit UI to upload resumes, paste job descriptions, and view analysis/matching/cover letter tabs.

**Backend (LangGraph)**:

- LangGraph `StateGraph` with four nodes: extract → analyze → match → recommend.
	- Extract node: pdfminer text extraction + LLM structuring of resume fields.
	- Analyze node: LLM summarizes skills/experience into a normalized schema.
	- Match node: LLM compares profile vs JD and returns title, match_score, and reason buckets.
	- Recommend node: LLM drafts a tailored cover letter.
- OrchestratorAgent builds and runs the graph; state carries resume path, job description, parsed data, match results, recommendation, status, and history.

**AI Integration**:

- Ollama (OpenAI-compatible endpoint, `model=llama3.2`) powers all node prompts.

---

## 📂 Folder Structure
- app.py: Streamlit application.
- agents/orchestrator.py: LangGraph pipeline definition and nodes.
- utils: Logging, exceptions.
- uploads: Uploaded resumes.
- results: Saved analysis outputs.

---

## 📝 How It Works
- Upload Resume: Users upload a PDF resume via the Streamlit interface.
- Input Job Description: Users paste a job description into the provided text area.

**AI Processing**:
- The resume is parsed, and the job description is analyzed.

**View Results** :
- Users can view skills analysis, job matches, screening scores, and a tailored cover letter.
- Save Results: Analysis outputs are saved in the results directory for future reference.

---
## 🚀 Getting Started
- Clone the repository.
- Install dependencies using pip install -r requirements.txt.
- Run the application with streamlit run app.py.
- Upload a resume and input a job description to get started!

---

## 📌 Future Enhancements
- Improve job matching accuracy with advanced NLP techniques.
- Allow editing of given resume in latex to fit job description for better chances
