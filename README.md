# Farmer Query Triage Dashboard — Quick Start

## What's in this folder
- `sample_queries.csv` — small placeholder dataset (60 rows, 6 categories) so everything runs immediately
- `train_model.py` — trains the classifier and saves `model.pkl` + `vectorizer.pkl`
- `model.pkl`, `vectorizer.pkl` — already trained on the sample data, so the dashboard works out of the box
- `app.py` — the Streamlit dashboard
- `requirements.txt` — the libraries you need

## Run it (takes 2 minutes)
```bash
pip install -r requirements.txt
streamlit run app.py
```
It opens automatically in your browser at `http://localhost:8501`.

## Before your final submission
1. Download the real KCC dataset from data.gov.in or the Kaggle mirror (links Claude gave earlier).
2. Save it as `sample_queries.csv` in this folder — just make sure the columns are named exactly `query_text` and `category` (rename the real dataset's columns if they're called something else, e.g. `QueryText`/`Sector`).
3. Re-run `python train_model.py` to retrain on the real data. This overwrites `model.pkl` and `vectorizer.pkl`.
4. Run `streamlit run app.py` again — the dashboard now reflects the real model, no code changes needed.

## If accuracy looks low
That's expected with only 60 sample rows — there aren't enough examples per category for the model to learn reliable word patterns yet. The real KCC dataset has far more rows per category, which should improve this a lot. If it's still not great after that, try:
- Merging very similar categories (e.g. combine niche subcategories into your 5-6 main ones)
- Trying `RandomForestClassifier` instead of `LinearSVC` in train_model.py and comparing accuracy
- Removing categories with very few examples

## Deploying for your viva (optional)
If you want a live link instead of running it locally during your presentation, [Streamlit Community Cloud](https://streamlit.io/cloud) hosts it for free — push this folder to a GitHub repo and connect it there.
