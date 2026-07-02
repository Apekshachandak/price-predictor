# Product Requirements Document — Tile Price Prediction




## 1. Problem Statement

A tile manufacturer sells approximately 2,840 ceramic and porcelain tiles (called SKUs). Each SKU has a specific set of attributes — physical dimensions, aesthetic properties, technical quality ratings, and a product photo. The manufacturer prices each tile in USD per square metre.

Given 2,272 SKUs with known prices (training set), build a model that accurately predicts the price of the remaining 568 SKUs (test set) from their attributes and photos.

---

## 2. What Success Looks Like

### Primary Metric
- **% of test SKUs predicted within ±10% of true price** — this is the metric the assignment specifically asks for. Maximise this.

### Secondary Metrics
- **Mean Absolute Error on actual price** — easy to explain to non-technical stakeholders ("we're off by $12 on average")
- **Root Mean Squared Error on log-price** — standard regression benchmark, penalises large errors more

### Qualitative Success
- The write-up clearly explains the data exploration findings
- Feature importance output makes intuitive sense (size and collection tier should dominate)
- Dead ends are documented honestly

---

## 3. Inputs

### Structured Data (`train.jsonl`, `test.jsonl`)
Each row is one SKU with these categories of fields:

| Category | Fields | Notes |
|---|---|---|
| Identity | `sku_id`, `collection_name`, `product_name` | Collection name is anonymised |
| Geometry | `product_width`, `product_length`, `product_thickness` | Nested objects with value + unit |
| Aesthetics | `color_family`, `finish_type`, `look_aesthetic`, `edge_type` | Categorical |
| Technical ratings | `pei_wear_rating`, `water_absorption_rate`, `breaking_strength`, `mohs_surface_hardness`, + many more | Nested objects, heavily missing |
| Packaging | `pieces_per_box`, `area_per_box`, `box_weight`, `boxes_per_pallet` | Plain numbers |
| Image path | `face_image` | Points to `images/<sku_id>.jpg` |
| **Target** | `price_usd_per_sqm` | Present in train, absent in test |

### Image Data (`images/`)
- One `.jpg` per SKU (train + test combined)
- Shows the tile face/surface texture
- High resolution, varied quality

---

## 4. Constraints

| Constraint | Detail |
|---|---|
| No external data | Only the provided JSONL and images. No web scraping, no external price databases. |
| Collection names anonymised | Can't look up real manufacturer pricing. Must infer from data. |
| Heavy missingness | Many technical fields are null for large subsets of SKUs. Must handle explicitly. |
| Small dataset | 2,272 training examples. Methods that need large data (deep neural nets from scratch) are inappropriate. |
| No GPU on local machine | Image embedding extraction needs GPU → use Kaggle free GPU tier. |

---

## 5. Solution Architecture

### Phase 1 — Structured Data Model
- Parse all nested JSON fields into flat numeric values
- Engineer derived features (tile area, log-price, collection aggregates, missingness flags)
- Train XGBoost gradient boosting model with 5-fold cross-validation
- Generate SHAP feature importance for explainability

### Phase 2 — Multimodal Enhancement
- Extract 512-dimensional visual embeddings from all tile photos using OpenAI's CLIP model (pre-trained, no fine-tuning)
- Reduce to 50 dimensions using Principal Component Analysis to avoid overfitting
- Append image features to structured feature table
- Retrain model, compare metrics

### Phase 3 — Analysis and Submission
- Select best model (with or without images based on validation score)
- Generate `predictions.csv`
- Write detailed analysis report

---

## 6. Feature Engineering Decisions

| Feature | How Built | Why |
|---|---|---|
| `tile_area_mm2` | `width × length` | Direct proxy for manufacturing cost and material quantity |
| `tile_volume_mm3` | `width × length × thickness` | Strongest single numeric predictor (r = 0.45 with log-price) |
| `aspect_ratio` | `length / width` | Long rectangular tiles price differently from squares |
| `finish_keyword` | Regex match on `product_name` | Pulido/Brillo/Mate/Antislip — more granular than structured `finish_type` |
| `app_combo` | Parsed from `application_location` list | Wall+Floor tiles command a premium over Wall-only |
| `log_price` | `log(price_usd_per_sqm)` | Normalises right-skewed target distribution |
| `collection_mean_price` | Mean price per collection (train only, fold-safe) | Collection is the primary quality-tier proxy |
| `collection_std_price` | Standard deviation per collection | Captures intra-collection price spread |
| `body_type` | Direct categorical | Color-Body vs. Red-Body carries meaningful price signal |
| `shade_variation_rating` | Direct categorical (V1–V4) | Higher variation tiles are systematically more expensive |
| `edge_type` | Direct categorical | Pressed/Cushioned vs. Rectified |
| `barefoot_val` | Ordinal from `barefoot_wet_slip_rating` | Class C (highest safety) commands a premium |
| `pendulum_val` | Ordinal from `pendulum_slip_class` | Slip resistance class as integer |
| `r_rating_val` | Integer extracted from `r_rating` | Surface grip rating (R9–R11) |
| `breaking_strength_val` | Numeric from nested object | r = 0.31 with log-price |
| `*_is_missing` flags | 1 if field is null, else 0 | Missingness itself is informative — budget tiles skip certification |
| `clip_pca_0` to `clip_pca_49` | CLIP embedding → PCA 50 dims | Visual aesthetics (marble/wood/concrete) not in structured data |

---

## 7. What Will NOT Be Done (and Why)

| Skipped | Reason |
|---|---|
| Fine-tuning CLIP on tile images | Needs more time and training data management; overkill for 3-day scope |
| One-hot encoding collection names | 200+ collections = 200+ sparse columns, causes overfitting on small dataset |
| Training a custom image regression network | Insufficient data, insufficient time, diminishing returns vs. pre-extracted features |
| Ensembling many models (blending 10+ models) | Marginal gain, disproportionate complexity for intern assignment |
| Hyperparameter grid search | Replaced by Optuna with 30 trials — faster and smarter |

---

## 8. Deliverables

| File | Description |
|---|---|
| `notebooks/01_EDA.ipynb` | Exploratory analysis with 8 charts and written observations |
| `notebooks/02_model.ipynb` | Tabular feature matrix, XGBoost 5-fold CV, SHAP analysis |
| `notebooks/03_final_submission.ipynb` | Side-by-side tabular vs. multimodal ablation, auto-selects best model |
| `notebooks/kaggle_clip_extraction.py` | GPU script for CLIP embedding extraction (run on Kaggle) |
| `outputs/submission.csv` | Predicted `price_usd_per_sqm` for all 568 test SKUs |
| `write_up.md` | Written narrative: what I found, what I tried, what I'd do next |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Collection leakage in target encoding | Medium | High | Compute collection stats inside CV folds only |
| CLIP embeddings don't improve score | **Confirmed** | Low | Structured model is the final submission; image experiment documented as honest finding |
| Test has new collections not in train | Low | Medium | Fall back to global mean for unknown collections |
| Kaggle GPU notebook crashes | Low | Medium | Save embeddings incrementally in batches |

---

## 10. Timeline

| Step | Estimated Time |
|---|---|
| Setup + data loading | 30 min |
| Nested field parsing | 1 hour |
| Feature engineering | 1.5 hours |
| EDA (5 charts + write-up) | 2 hours |
| Baseline model + cross-validation | 1.5 hours |
| SHAP explainability | 30 min |
| CLIP embedding extraction (Kaggle) | 1 hour (includes upload + run + download) |
| Add image features + retrain | 30 min |
| Final predictions + submission file | 30 min |
| Write-up | 1.5 hours |
| **Total** | **~10.5 hours** |
