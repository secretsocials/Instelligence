
# Instagram Analytics Dashboard (Streamlit)

A data-driven Instagram analysis & scheduling toolkit. Upload your multi-sheet Excel export and get:
- Engagement metrics (post-level & profile-level)
- Caption theme clustering + performance
- Hashtag performance (incl. hashtags vs no-hashtags by post type)
- Posting time heatmap (day × hour)
- Auto-generated 90‑day posting calendar (Excel, CSV, ICS)

## 🧱 Project Structure
```
.
├── ig_analytics_pipeline.py   # Core analysis functions + CLI
├── streamlit_app.py           # Streamlit UI
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start (Local)
```bash
# 1) Clone or download this repo
cd ig-analytics-streamlit-repo

# 2) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run streamlit_app.py
```
Open the URL Streamlit prints (usually http://localhost:8501), upload your Excel export, choose options, and click **Run Analysis**.

## ☁️ Deploy to Streamlit Community Cloud (Recommended)
1. Push this folder to a **public GitHub repo**.
2. Go to **https://share.streamlit.io** → Sign in → **New app**.
3. Select your repo & branch.
4. Set **Main file path** to `streamlit_app.py`.
5. Click **Deploy**.

Streamlit Cloud will install from `requirements.txt` and build automatically. You’ll get a public URL to use on desktop and phone.

## 🧪 Accepted Input
- One Excel file (`.xlsx`) with **one sheet per profile**.
- Each sheet should contain a **`Followers`** row and a **`Post #`** table as in the provided template.

## 🧰 CLI Usage (Optional)
You can also run the pipeline headless:
```bash
python ig_analytics_pipeline.py   --input /path/to/instagram_data.xlsx   --start-date 2025-10-01   --output ./output   --tz Europe/London
```
Outputs include CSVs, a heatmap PNG, and (if `--start-date` is passed) a 90‑day calendar: `.xlsx`, `.csv`, `.ics`.

## 🔧 Configuration
- Timezone: choose in the Streamlit sidebar (or `--tz` for CLI).
- Calendar start date: choose in UI (or `--start-date` for CLI).

## 🐞 Troubleshooting
- If Streamlit doesn’t open a browser, manually visit `http://localhost:8501`.
- On Apple Silicon, upgrading build tools helps:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```
- If a dependency fails to build, ensure Python ≥ 3.9.

## 📄 License
MIT
