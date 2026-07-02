# PRD → Issues Breakdown

> Breaking the PRD into concrete, trackable tasks. Each issue is one self-contained unit of work with a clear definition of done.

---

## ISSUE #1 — Project Setup
**Status:** `[x] Done`
**Estimated Time:** 30 minutes

### What to do
- Create folder structure: `notebooks/`, `embeddings/`, `outputs/`, `brainstorm/`
- Install all required Python packages
- Start Jupyter Notebook and verify it opens

### Packages to install
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap Pillow optuna
```

### Definition of Done
- All folders exist
- Running `import xgboost` in a notebook cell shows no errors
- Jupyter is running at `localhost:8888`

---

## ISSUE #2 — Load Raw Data
**Status:** `[x] Done`
**Estimated Time:** 30 minutes

### What to do
- Create `notebooks/01_EDA.ipynb`
- Load `data/train.jsonl` into a pandas dataframe (2,272 rows expected)
- Load `data/test.jsonl` into a pandas dataframe (568 rows expected)
- Print column names, data types, and first 3 rows of each

### Definition of Done
- `train_df.shape` returns `(2272, ~40)`
- `test_df.shape` returns `(568, ~40)`
- No errors during loading

---

## ISSUE #3 — Parse Nested Fields
**Status:** `[x] Done`
**Estimated Time:** 1 hour

### What to do
- Write `extract_nested_value(field)` function that pulls the `value` key out of nested dicts
- Write `extract_comparator(field)` function that returns 1 if comparator is `=`, 0 if `<` or `>`
- Apply both functions to all nested fields in both `train_df` and `test_df`
- Fields to parse: `product_width`, `product_length`, `product_thickness`, `water_absorption_rate`, `breaking_strength`, `modulus_of_rupture`, `mohs_surface_hardness`, `recommended_grout_joint`, `pei_wear_rating`, `linear_thermal_expansion`

### Definition of Done
- `train_df["product_width_value"]` is a column of floats (or NaN where missing)
- `train_df["product_width_is_exact"]` is a column of 0s and 1s (or NaN)
- Same columns exist in `test_df`

---

## ISSUE #4 — Feature Engineering
**Status:** `[x] Done`
**Estimated Time:** 1.5 hours

### What to do

**Derived geometric features:**
- `tile_area_mm2 = product_width_value × product_length_value`
- `tile_volume_mm3 = tile_area_mm2 × product_thickness_value`

**Target transformation:**
- `log_price = log(price_usd_per_sqm)` — only for train

**Collection-level features (compute on train only, then join to test):**
- `collection_mean_price` — mean price per collection
- `collection_std_price` — standard deviation of price per collection
- `collection_count` — number of SKUs per collection

**Missingness flags:**
- For each important technical field, add a binary column `*_is_missing` = 1 if null, 0 if present
- Fields to flag: `pei_value`, `water_absorption_rate_value`, `breaking_strength_value`, `mohs_surface_hardness_value`

### Definition of Done
- `train_df["tile_area_mm2"]` exists and has no zeros (only NaN where inputs were missing)
- `train_df["log_price"]` exists and has no -inf values
- `train_df["collection_mean_price"]` exists and matches known collection averages
- `train_df["pei_value_is_missing"]` is 1 where `pei_value` is NaN, 0 otherwise

---

## ISSUE #5 — Exploratory Data Analysis (5 Charts)
**Status:** `[x] Done`
**Estimated Time:** 2 hours

### What to do
Create the following 5 charts in `01_EDA.ipynb`. Each chart must have a markdown cell below it explaining what you see in plain English.

| Chart | What to plot | What to look for |
|---|---|---|
| 1 | Price distribution (raw vs. log) — side-by-side histograms | Right skew in raw; discrete pricing tiers visible in log |
| 2 | Tile volume vs. price — scatter plot | Positive correlation (r = 0.45); vertical stacks show other factors matter too |
| 3 | Price by finish keyword — box plots | Pulido/Brillo/Mate/Antislip price tiers |
| 4 | Full numeric correlation bar chart | Which features correlate with log-price? |
| 5 | Price by: body type, shade variation, edge type, barefoot slip — 2x2 grid of box plots | Clear price signal from categorical attributes |

Save all charts as `.png` files in `outputs/`.

### Definition of Done
- 5 charts rendered without errors
- Each chart has a written explanation below it (at least 2 sentences)
- Charts are saved to `outputs/`

---

## ISSUE #6 — Build Final Feature Matrix
**Status:** `[x] Done`
**Estimated Time:** 45 minutes

### What to do
- Define `categorical_cols` list and `numeric_cols` list
- Use `LabelEncoder` from scikit-learn to convert categorical columns to integers
- Combine train and test before encoding so labels are consistent across both
- Build `X_train` (feature matrix) and `y_train` (log price target)
- Build `X_test` (feature matrix for prediction)
- Fill all remaining NaN values with `-999` (XGBoost handles this as "missing")

### Definition of Done
- `X_train` has shape `(2272, N_features)` with no NaN values
- `X_test` has shape `(568, N_features)` with no NaN values
- All columns in `X_train` and `X_test` are numeric (float or int)

---

## ISSUE #7 — Train XGBoost with Cross-Validation
**Status:** `[x] Done`
**Estimated Time:** 1.5 hours

### What to do
Create `notebooks/02_model.ipynb`. In it:
- Set up 5-fold cross-validation using `KFold` from scikit-learn
- In each fold: train XGBoost, predict on validation fold, predict on test
- Collect out-of-fold predictions for training rows
- Average test predictions across all 5 folds
- Convert log-price predictions back to actual price using `exp()`

**XGBoost parameters to start with:**
```python
n_estimators=1000
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
early_stopping_rounds=50
```

**Report after training:**
- Mean Absolute Error (actual price)
- Root Mean Squared Error (log price)
- % predictions within ±10% of true price

### Definition of Done
- Model trains without error
- Cross-validation score is reported for all 3 metrics
- `test_predictions_log` array has 568 values

---

## ISSUE #8 — SHAP Feature Importance
**Status:** `[x] Done`
**Estimated Time:** 30 minutes

### What to do
- Use `shap.TreeExplainer(model)` on the last trained fold's model
- Generate `shap_values` for the training set
- Create a bar chart showing top 15 most important features
- Save chart to `outputs/shap_importance.png`
- Write a paragraph explaining what the top features mean

### Definition of Done
- SHAP bar chart rendered without error
- `tile_area_mm2` and `collection_mean_price` appear in the top 5 (they should)
- Written explanation is in the notebook

---

## ISSUE #9 — CLIP Image Embedding Extraction (Kaggle GPU)
**Status:** `[x] Done` — **Manual step, run on Kaggle**
**Estimated Time:** 1 hour (includes upload, run, download)

### What to do
1. Go to kaggle.com → Create New Notebook
2. Settings → Accelerator → GPU T4 x2
3. Upload the `images/` folder as a Kaggle Dataset
4. Paste the extraction code from `implementation_plan.md` Step 9
5. Run the notebook (~10 minutes)
6. Download `clip_embeddings.npy` and `clip_sku_ids.npy`
7. Place both files in `tile_pricing_challenge/embeddings/`

### What CLIP does
CLIP is a model pre-trained on hundreds of millions of image-text pairs. Given a tile photo, it produces a 512-number vector that captures the visual "character" of the tile — whether it looks like marble, wood, concrete, whether it's intricate or plain. We use these numbers as additional features.

### Definition of Done
- `embeddings/clip_embeddings.npy` exists locally
- `embeddings/clip_sku_ids.npy` exists locally
- Shape of `clip_embeddings.npy` is `(2840, 512)`

---

## ISSUE #10 — Add Image Features to Model
**Status:** `[x] Done`
**Estimated Time:** 45 minutes

### What to do
- Load `clip_embeddings.npy` and `clip_sku_ids.npy`
- Build a lookup dictionary: `sku_id → 512-number vector`
- Align embeddings to the order of `train_df["sku_id"]` and `test_df["sku_id"]`
- Apply PCA (Principal Component Analysis) to reduce 512 dimensions to 50
  - Fit PCA on training embeddings only
  - Transform both train and test using the same PCA
- Add 50 new columns (`clip_pca_0` ... `clip_pca_49`) to `X_train` and `X_test`
- Retrain XGBoost with the same cross-validation setup
- Compare metrics: did images help?

### Definition of Done
- `X_train_with_images` has shape `(2272, N_features + 50)`
- Model retrained without error
- Metrics table shows before-vs-after comparison

---

## ISSUE #11 — Generate Submission File
**Status:** `[x] Done`
**Estimated Time:** 30 minutes

### What to do
- Create `notebooks/03_submission.ipynb`
- Choose the best model (with or without images, based on cross-validation score)
- Take the averaged test predictions (log-price), convert back to actual price with `exp()`
- Create a dataframe with two columns: `sku_id` and `price_usd_per_sqm`
- Verify: 568 rows, no NaN values, all prices are positive and reasonable (e.g., between $5 and $500)
- Save to `outputs/predictions.csv`

### Definition of Done
- `outputs/predictions.csv` exists with 568 rows
- No NaN or negative values
- File has exactly two columns: `sku_id` and `price_usd_per_sqm`

---

## ISSUE #12 — Write the Analysis Write-Up
**Status:** `[x] Done`
**Estimated Time:** 1.5 hours

### What to do
Create `write_up.md` in the root of the project folder. Structure it as:

1. **What I found in the data** — key EDA findings, what surprised you
2. **Features I engineered and why** — tile area, collection stats, missingness flags
3. **Model choice and training setup** — why XGBoost, what cross-validation means and why you used it
4. **Results** — table of metrics for baseline and image-enhanced model
5. **What worked and what didn't** — honest assessment
6. **What I would do with more time** — fine-tuning CLIP, neural collection embeddings, etc.

### Tone
Write as if explaining to a technical teammate who wasn't there. Be specific — mention actual numbers from your results. Don't be vague ("the model performed well"). Say "the model predicted 68% of test tiles within ±10% of true price."

### Definition of Done
- `write_up.md` exists in the project root
- All 6 sections are filled in with specific numbers
- At least 500 words

---

## Progress Tracker

| Issue | Description | Status |
|---|---|---|
| #1 | Project Setup | `[x]` |
| #2 | Load Raw Data | `[x]` |
| #3 | Parse Nested Fields | `[x]` |
| #4 | Feature Engineering | `[x]` |
| #5 | EDA (8 Charts) | `[x]` |
| #6 | Build Feature Matrix | `[x]` |
| #7 | Train XGBoost with Cross-Validation | `[x]` |
| #8 | SHAP Feature Importance | `[x]` |
| #9 | CLIP Embedding Extraction (Kaggle) | `[x]` |
| #10 | Add Image Features + Retrain | `[x]` — images degraded accuracy; tabular-only model selected |
| #11 | Generate Submission File | `[x]` |
| #12 | Write Analysis Write-Up | `[x]` |
