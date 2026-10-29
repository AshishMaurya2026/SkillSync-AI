# SkillSync AI — Resume Analyzer

**Team:** Ashish Maurya · Aryan Sachan · Aviral Singh · Aryan Verma

A student mini-project that extracts text from a PDF resume, detects skills using a curated dictionary, compares them with a target job description, checks common resume sections, and generates improvement suggestions. It also calculates TF-IDF cosine similarity as a text-overlap indicator (not an official ATS score).

## Features
- PDF resume upload and text extraction (PyMuPDF)
- Resume/job skill matching and possible skill gaps
- Potential missing keywords
- Resume section checks
- Text similarity indicator (clearly labelled, not a validated ATS score)
- Downloadable analysis summary
- Dark navy/cyan interface with high-contrast text
- Optional Gemini API-powered suggestions; the app still works without an API key

## Run locally on Windows
1. Extract the ZIP.
2. Open the extracted `SkillSync_AI_Final_Project` folder.
3. Click the address bar in File Explorer, type `cmd`, and press Enter.
4. Run these commands one by one:

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

5. Open `http://localhost:8501` if the browser does not open automatically. Keep the black terminal window open while using the app.

## Optional Gemini AI suggestions
The built-in resume analysis works without a Gemini key. To enable generative suggestions:
1. Create an API key in Google AI Studio: https://aistudio.google.com/apikey
2. **Do not put the key in `app.py` or commit it to GitHub.**
3. For local use, create `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "paste-your-key-here"
```

4. Restart the app. This file is excluded by `.gitignore`.

For Streamlit Community Cloud, open the app's **Settings → Secrets** and add the same line there.

## Publish with GitHub + Streamlit Community Cloud
1. Sign in to GitHub and create a **public** repository named `skillsync-ai-resume-analyzer` (or another name you prefer).
2. Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, and the `.streamlit/config.toml` file. Do not upload `.venv`, API keys, or `secrets.toml`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository, branch `main`, and main file path `app.py`; deploy.
5. When the build finishes, open the provided `*.streamlit.app` URL and test a PDF upload and job description.
6. If using Gemini, add the API key under the deployed app's Settings → Secrets, then reboot the app.

## Test checklist
- Upload `sample_resume_ai_ml_intern.pdf`.
- Paste a target role description with Python, machine learning, SQL, Pandas, NumPy, scikit-learn and Git.
- Click **Analyse resume** and check skills, matches, gaps and section checks.
- Download the analysis report.
- Try an empty upload and a short job description to confirm validation messages.
- Test a different PDF and job description.

## Project limitations
- Skill matching uses a finite curated dictionary; related or differently worded skills may be missed.
- Scanned/image-only PDFs are not OCR-processed.
- TF-IDF similarity is a text-overlap measure, not an official ATS score, hiring probability or guarantee of suitability.
- Gemini suggestions are optional and depend on a valid API key, network access and model availability.
- Recommendations are advisory. Never claim a skill or experience you do not have.
