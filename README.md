# IPL Win Predictor

I built a real-time IPL win-probability model by converting raw ball-by-ball data into meaningful match-state features like runs-left, balls-left, wickets, CRR and RRR. I cleaned and joined the Kaggle datasets, engineered context features, and trained a logistic regression pipeline using ColumnTransformer. The final model was packaged as pipe.pkl, so it can power dashboards or live simulations. This project shows full end-to-end ownership—from data engineering to modeling and deployment-ready inference.”

## Project Highlights (Interview Talking Points)
- End‑to‑end ETL inside the notebook: data cleaning, feature engineering, model training, evaluation, and visualization in one place.
- Purpose‑built match state features (`runs_left`, `balls_left`, `wickets`, `crr`, `rrr`) paired with categorical context (`batting_team`, `bowling_team`, `city`) to capture pressure dynamics.
- Reusable `sklearn` pipeline with `ColumnTransformer` + `LogisticRegression`; saved via `pickle` for deployment or dashboards.
- Scenario visualizer that plots win/loss probability trajectories over overs (see `match_progression` helper near the end of the notebook).

## Repository Layout
- `Untitled.ipynb` – main notebook covering exploration, preprocessing, modeling, and plotting.
- `matches.csv` / `deliveries.csv` – Kaggle IPL datasets (through 2019) used for training.
- `pipe.pkl` – trained scikit‑learn pipeline ready for inference.

## Data Preparation & Feature Engineering
1. Filter out abandoned/DL-affected games and keep the eight most recent franchises (e.g., map Delhi Daredevils → Delhi Capitals, Deccan Chargers → Sunrisers Hyderabad).
2. Join first‑innings totals with second‑innings deliveries.
3. Derive match‑state columns:
   - `runs_left`, `balls_left` (target chase context)
   - `wickets` (remaining wickets computed from cumulative dismissals)
   - `crr` (current run rate) and `rrr` (required run rate)
4. Drop incomplete rows (`NaN`, zero balls left) and shuffle the dataset before splitting.

## Modeling
- Split: 80/20 train-test via `train_test_split`.
- Transformer: `ColumnTransformer` with `OneHotEncoder(drop='first', sparse_output=False)` on team + city columns; numerical columns passed through.
- Estimator: `LogisticRegression(solver='liblinear')` wrapped in an `sklearn.Pipeline`.
- Metric: plain accuracy (`accuracy_score`) for quick benchmarking; more nuanced metrics (log-loss, calibration) can be added later.

## Getting Started
1. **Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt  # or install pandas numpy scikit-learn matplotlib
   ```
2. **Data** – ensure `matches.csv` and `deliveries.csv` sit beside the notebook.
3. **Notebook** – open `Untitled.ipynb`, run all cells to reproduce the pipeline and plots.
4. **Model Artifact** – re-running the notebook overwrites `pipe.pkl` with the latest pipeline.

## Using the Trained Model
Load the pickle and pass a single-row DataFrame that mirrors the training schema:

```python
import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))

state = pd.DataFrame([{
    'batting_team': 'Chennai Super Kings',
    'bowling_team': 'Mumbai Indians',
    'city': 'Mumbai',
    'runs_left': 45,
    'balls_left': 30,
    'wickets': 6,
    'total_runs_x': 175,  # target
    'crr': 8.3,
    'rrr': 9.0
}])

win_prob = pipe.predict_proba(state)[0][1]
print(f'Win probability: {win_prob:.2%}')
```

## Visualizing Win Trajectories
The helper `match_progression(delivery_df, match_id, pipe)` builds over-by-over win/lose curves and bar charts. Use it to showcase insights (“Here’s how KKR’s chances flipped after the 15th over...”) during interviews.

## Extension Ideas
- Swap in gradient boosting or sequence models for potentially higher accuracy.
- Add weather/venue bias features or toss information.
- Deploy the pickle behind a FastAPI/Streamlit interface for live demos.
- Track calibration and Brier score to describe probabilistic quality.

## How to Talk About It
1. **Problem framing** – “Given the live state of a chase, estimate the batting side’s win probability.”
2. **Data engineering** – Outline how you transformed ball-by-ball logs into summarized per-delivery features.
3. **Modeling choices** – Stress the interpretability and speed of logistic regression with proper encoding.
4. **Results & demo** – Mention accuracy (from the notebook) and show the probability curve visualization.
5. **Next steps** – Discuss extension ideas above to demonstrate product thinking.

Use this README as a guide when walking interviewers through the project, focusing on the clear data-to-model narrative, reproducibility via the notebook, and the saved artifact that proves you can move from exploration to deployable assets.

