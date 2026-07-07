# House Price Predictor (Streamlit)

Linear Regression model trained on the Kaggle "Housing Prices Dataset"
(545 listings). Two tabs: **Predict** a price from house details, and
**Insights** into the data and what drives the model's predictions.

## Files
- `app.py` — the Streamlit app (trains the model in-memory on startup, cached)
- `Housing.csv` — the dataset (bundled so the app is self-contained)
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL it prints (usually http://localhost:8501).

## Deploy to Streamlit Community Cloud (free, public URL)
1. Create a new GitHub repo and push these three files to it
   (`app.py`, `Housing.csv`, `requirements.txt`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick your repo, branch `main`, and set
   **Main file path** to `app.py`.
4. Click **Deploy**. First build takes 1-2 minutes; you'll get a public
   URL like `https://<yourapp>.streamlit.app`.

## Notes
- Model: plain Linear Regression, R² ≈ 0.65 on held-out data
  (typical error ± ₹9.8L on prices averaging ~₹47.7L). It's a solid,
  interpretable baseline — not a highly precise valuation tool.
- `furnishingstatus` keeps all 3 original categories (furnished /
  semi-furnished / unfurnished) via label encoding, unlike the original
  GitHub notebook which collapsed semi-furnished into furnished and lost
  information.
- Everything retrains from `Housing.csv` on app startup (cached via
  `st.cache_resource`), so there's no separate model file to keep in sync —
  simpler to maintain and redeploy.
