# create_eduscan.py — FIXED & COMPLETE

from pathlib import Path
import json

ROOT = Path('eduscan-mvp')
ROOT.mkdir(exist_ok=True)


def write_file(rel_path: str, content: str) -> None:
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')
    print("\nProject generation complete: eduscan-mvp/")

# README
README = """# EduScan - MVP (Streamlit-ready)

This project is an MVP for automatic grading of multiple-choice answer sheets.

Quick start:
1. pip install -r requirements.txt
2. streamlit run ui/streamlit_app.py

Deploy to Streamlit Cloud: point entry file to ui/streamlit_app.py
"""
write_file('README.md', README)

REQS = """streamlit
Pillow
numpy
pandas
openpyxl
requests
fastapi
uvicorn
pyinstaller
"""
write_file('requirements.txt', REQS)

# Sample answer key
AK = {str(i+1): "A" for i in range(10)}
write_file('sample_data/answer_key.json', json.dumps(AK, indent=2))

# Backend
BACKEND = r"""from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uuid, shutil, os

app = FastAPI()
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/upload-image')
async def upload_image(file: UploadFile = File(...), test_id: str = 'default'):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f'{file_id}_{file.filename}')
    with open(path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({'status': 'ok', 'path': path})
"""
write_file('app/backend.py', BACKEND)

# Utils
UTILS = r"""import pandas as pd

def export_results_to_excel_local(results, out_path='results.xlsx'):
    rows = []
    for r in results:
        rows.append({
            'STT': r.get('stt', ''),
            'Score': r.get('score', 0),
            'Confidence': r.get('confidence', 0),
            'Answers': ','.join([f"{k}:{v}" for k,v in r.get('answers', {}).items()])
        })
    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    return out_path
"""
write_file('app/utils.py', UTILS)

# DB
DB = """def init_db():
    return

def save_submission(*args, **kwargs):
    return None

def get_test_results(test_id):
    return []
"""
write_file('app/db.py', DB)

# Streamlit UI
STREAMLIT_APP = r"""import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import json
import pandas as pd
from app.utils import export_results_to_excel_local

st.set_page_config(page_title='EduScan - Auto Grader', layout='wide')
st.title('EduScan — Simple OMR Grader (Streamlit)')

with st.sidebar:
    st.header('Settings')
    num_questions = st.number_input('Number of questions', 1, 200, 10)
    num_columns = st.number_input('Options per question', 2, 6, 4)
    key_text = st.text_input('Answer key (space-separated)')
    sample_key = st.checkbox('Use sample answer_key.json', value=True)

uploaded = st.file_uploader('Upload student sheet', type=['png','jpg','jpeg'])

if key_text.strip():
    answer_key = key_text.strip().upper().split()
elif sample_key:
    try:
        with open('sample_data/answer_key.json','r', encoding='utf-8') as f:
            answer_key_json = json.load(f)
            answer_key = [answer_key_json.get(str(i),'N') for i in range(1, num_questions+1)]
    except:
        answer_key = ['N'] * num_questions
else:
    answer_key = ['N'] * num_questions


def image_to_np(img_file):
    image = Image.open(img_file).convert('RGB')
    return np.array(ImageOps.exif_transpose(image))


def simple_omr_detect(img_np, num_q, opts_per_q):
    h, w = img_np.shape[:2]
    gray = (0.299*img_np[:,:,0] + 0.587*img_np[:,:,1] + 0.114*img_np[:,:,2]).astype(np.uint8)
    th = (gray < 128).astype(np.uint8) * 255

    answers = {}
    confidences = []
    row_h = h // num_q

    for i in range(num_q):
        y0, y1 = i*row_h, (i+1)*row_h
        row = th[y0:y1, :]
        col_w = w // opts_per_q
        fractions = []
        for j in range(opts_per_q):
            x0, x1 = j*col_w, (j+1)*col_w if j < opts_per_q-1 else w
            cell = row[:, x0:x1]
            frac = (cell > 0).sum() / (cell.size + 1e-6)
            fractions.append(frac)
        choice = int(np.argmax(fractions))
        choice_map = [chr(ord('A')+k) for k in range(opts_per_q)]
        answers[str(i+1)] = choice_map[choice]
        confidences.append(float(fractions[choice]))

    return answers, confidences


if uploaded:
    st.image(uploaded)
    img_np = image_to_np(uploaded)
    answers, conf = simple_omr_detect(img_np, int(num_questions), int(num_columns))

    rows = []
    correct = 0
    for i in range(1, num_questions+1):
        d = answers.get(str(i), 'N')
        k = answer_key[i-1] if i-1 < len(answer_key) else 'N'
        ok = (d == k)
        correct += ok
        rows.append({'Question': i, 'Detected': d, 'Answer': k, 'Correct': ok, 'Confidence': round(conf[i-1],3)})

    st.table(pd.DataFrame(rows))
    st.metric('Score', f'{correct}/{num_questions}')

    if st.button('Export to Excel'):
        payload = [{'stt':'', 'answers':answers, 'score':correct, 'confidence':float(np.mean(conf))}]
        out = 'results_student.xlsx'
        export_results_to_excel_local(payload, out)
        with open(out, 'rb') as f:
            st.download_button('Download', f, file_name=out)
else:
    st.info('Upload a student sheet to begin.')
"""
write_file('ui/streamlit_app.py', STREAMLIT_APP)

# start_app.bat
START_BAT = r"""@echo off
start "EduScan Backend" cmd /k "uvicorn app.backend:app --reload --port 8000"
start "EduScan UI" cmd /k "streamlit run ui/streamlit_app.py"
"""
write_file('start_app.bat', START_BAT)

# build_exe
BUILD_EXE = r"""@echo off
python -m PyInstaller --onefile --noconfirm ui/streamlit_app.py
"""
write_file('build_exe.bat', BUILD_EXE)

# Dockerfile
DOCKERFILE = r"""FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0"]
"""
write_file('Dockerfile', DOCKERFILE)

# docker-compose
docker_compose = """services:
  eduscan:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/app
"""
write_file('docker-compose.yml', docker_compose)

# tests
TESTS = """import json

def test_answer_key_load():
    with open('sample_data/answer_key.json') as f:
        data = json.load(f)
    assert data.get('1') == 'A'
"""
write_file('tests/test_basic.py', TESTS)

print("
Project generation complete: eduscan-mvp/
")
