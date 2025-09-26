# Instagram Analytics Dashboard (Streamlit Cloud Ready)

Upload a multi‑sheet Instagram Excel export and get:
- Engagement metrics (post- and profile-level)
- Caption theme clustering + performance
- Hashtag performance (incl. hashtags vs no-hashtags by post type)
- Posting time heatmap (day × hour)
- Auto-generated 90‑day posting calendar (Excel, CSV, ICS)

## Deploy on Streamlit Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io → **New app**.
3. Select repo/branch and set **Main file path** to `streamlit_app.py` → **Deploy**.
4. Upload your `.xlsx` in the app and hit **Run Analysis**.

`requirements.txt` includes **openpyxl** for Excel support, and `.streamlit/config.toml` disables file watching warnings.

## Local Run (optional)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Input Format
One Excel file (`.xlsx`) with **one sheet per profile**. Each sheet should contain:
- a `Followers` row (followers value in the 2nd column)
- a `Post #` header row that starts a 9‑column posts table:
  `Post # | Caption | Likes | Comments | Upload Date | Upload Time | Post Type | Media Link | Hashtags`
