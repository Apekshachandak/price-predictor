# Tile Price Prediction — Write-Up




## How I Approached This

Before writing any code, I spent time brainstorming and stress-testing my plan. I structured this as a three-phase process — each phase documented in the `brainstorm/` folder:

1. **Grill Me** (`brainstorm/01_grill_me.md`) — I interrogated every assumption: why would a model work here, what's the biggest risk (collection leakage), why XGBoost over a neural net, why log-price, what if images don't help? The goal was to catch bad decisions before they cost time.

2. **Product Requirements** (`brainstorm/02_prd.md`) — I formalised the feature engineering plan: which fields to parse, what to engineer, how to handle missingness, and how to use collection statistics without memorising them.

3. **Issues Breakdown** (`brainstorm/03_prd_to_issues.md`) — I broke the work into concrete tasks with acceptance criteria, so I could track progress and avoid scope creep.

This upfront thinking shaped every downstream decision. Most of the "hard" work happened here — the code was just execution.

---

## What I Found in the Data

### Price Distribution
The 2,272 training tiles range from **$5.96 to $33.56/m²**, with a median around **$12.74/m²**. The distribution is right-skewed, and viewing it on a log scale reveals discrete pricing clusters at fixed intervals. This isn't a smooth market — it's structured manufacturer pricing, where tiles are placed into deliberate brackets tied to product tiers.

Training on `log(price)` and converting back at inference time normalises this skewness and ensures the model treats proportional errors fairly. Being off by $2 on a $10 tile is a bigger mistake than being off by $2 on a $30 tile — log-transformation encodes that intuition naturally.

### Size Is the Strongest Lever
The README hinted that size drives per-m² price, and the data confirmed it emphatically. Each tile's dimensions are stored as nested JSON objects (`{"value": 1200.0, "comparator": "=", "unit": "mm"}`), which required careful parsing. From the extracted width, length, and thickness, I engineered:

- **Tile volume** (`width × length × thickness`) — Pearson correlation **r = 0.45** with log-price, the strongest single numeric feature. Bigger, thicker tiles cost more to manufacture, are heavier to ship, and break more easily in transit.
- **Tile area** (`width × length`) — r = 0.40. Volume beat area because the thickness dimension carries real signal: a 6mm slab-format tile and a 20mm outdoor paver have very different cost structures.
- **Aspect ratio** (`length / width`) — captures format differences. Long, narrow plank tiles (simulating wood) price differently from square tiles of the same area.

### Missingness Is Structural, Not Random
The dataset has 60+ fields, and many technical ratings (PEI wear, slip resistance, water absorption) are null for 30–65% of products. This isn't data quality noise — it's a structural signal. Budget tiles don't go through certification labs. When a tile has no PEI rating and no slip test result, it's almost certainly a basic wall tile, not a premium floor product.

Rather than imputing missing values (which would inject false information), I created boolean `_is_missing` flags for each technical field. XGBoost can then learn the rule: "if PEI rating is missing AND water absorption is missing, this tile is probably in the $8–$10 bracket." These flags turned out to be among the top features by SHAP importance.

### Collection Carries Enormous Signal — Used Carefully
Tiles within the same collection share materials, finishes, and target markets. Their prices cluster tightly — most collections have a within-collection coefficient of variation below 0.05. The collection is effectively a product-line quality tier.

But there are ~200 collections. One-hot encoding would be wasteful, and label-encoding the name as an integer carries no meaning. Instead, I used target encoding: each collection is represented by its **mean price** and **standard deviation**, computed **exclusively from training data within each cross-validation fold**. This prevents the model from cheating by looking at validation prices during training. `collection_mean_price` became the single most important feature in SHAP analysis — which makes intuitive sense: knowing which product line a tile belongs to tells you most of what you need to know about its price tier.

### Finish Keywords from Product Names
The structured `finish_type` field only has three values (Matte, Glossy/Polished, and null). But the product name follows a consistent pattern — `[Collection] [Color] [Finish] [Size] [Extras]` — and encodes much richer finish information. A regex extraction yields keywords like `Pulido` (polished), `Brillo` (glossy), `Mate` (matte), and `Antislip`, each with clean price separation: Pulido tiles median at **$17.38/m²** vs. Mate at **$11.71/m²**.

### Categorical Features With Real Signal
A full audit of all 60+ fields surfaced several categorical features that meaningfully separate price tiers:

| Feature | Finding |
|---|---|
| **Body type** | Color-Body ($13.72 median) > Neutral-Body ($10.85) > White-Body ($10.56) > Red-Body ($5.96). Clay composition directly affects material cost. |
| **Shade variation** | V3/V4 (high pattern variation) tiles are more expensive. Complex surface patterns require more manufacturing precision. |
| **Edge type** | Pressed/Cushioned edges ($19.19 median) vs. Rectified ($12.57). Older decorative formats command premiums in niche segments. |
| **Barefoot slip rating** | Class C (highest grip, $20.34) vs. Class A ($11.71). High safety ratings indicate premium commercial-grade tiles. |
| **Application location** | Wall+Floor tiles (dual-purpose, more durable) command a premium over Wall-only tiles. |

---

## Features Engineered

| Feature | How It's Built | Why It Matters |
|---|---|---|
| `tile_volume_mm3` | width × length × thickness | Strongest numeric predictor (r = 0.45); captures material mass |
| `tile_area_mm2` | width × length | Direct manufacturing cost proxy |
| `aspect_ratio` | length / width | Plank vs. square format pricing |
| `finish_keyword` | Regex from `product_name` | More granular than `finish_type` (Pulido/Brillo/Mate/Antislip) |
| `app_combo` | Parsed `application_location` list | Wall+Floor vs. Wall-only |
| `collection_mean_price` | Mean log-price per collection (fold-safe) | Single most important feature by SHAP |
| `collection_std_price` | Std deviation per collection | Captures intra-collection variability |
| `*_is_missing` flags | 1 if field is null, 0 if present | Missingness correlates with product tier |
| `body_type`, `shade_variation_rating`, `edge_type` | Direct categoricals | Each separates price tiers clearly |
| `barefoot_val`, `pendulum_val`, `r_rating_val` | Ordinal from class labels | Slip resistance as ordered integers |
| `clip_pca_0..49` | CLIP ViT-B/32 embedding → PCA 50 dims | Visual features (texture, color, pattern) |

---

## Model and Training

### Why XGBoost
With 2,272 training rows, neural networks would overfit without heavy regularisation. XGBoost is built for exactly this regime: small-to-medium tabular data with missing values. It handles NaNs natively (learning the optimal split direction for missing values), trains in seconds, and produces SHAP-interpretable predictions. For an assignment where showing your reasoning matters as much as accuracy, interpretability is a feature, not a nice-to-have.

### Cross-Validation Strategy
5-fold cross-validation with a fixed random seed. Each fold:
1. Computes collection statistics (mean, std, count) from **training rows only** — never touching the validation set.
2. Trains an XGBoost model with early stopping (patience = 50 rounds).
3. Records out-of-fold predictions for the validation set and averages test predictions across all 5 folds.

This gives an honest performance estimate on data the model has never seen, without touching the test set.

### Image Embeddings — The Experiment
The README noted that photos encode visual information (marble vs. concrete vs. wood, pattern richness, colour) that structured fields only partly capture. I ran OpenAI's pre-trained CLIP model (ViT-B/32) on all 2,840 tile images using a Kaggle GPU notebook, extracting 512-dimensional visual embeddings. PCA compressed these to 50 dimensions (capturing 84.9% of the visual variance) to avoid overwhelming the 43 tabular features.

The extraction script (`notebooks/kaggle_clip_extraction.py`) processes images in batches of 64 and normalises embeddings to unit length — standard practice with CLIP.

---

## Results

I ran both models — tabular-only and tabular + vision — using identical cross-validation splits, and compared them side by side:

| Metric | Tabular Only | Tabular + Vision |
|---|---|---|
| Mean Absolute Error | **$0.11** | $0.28 |
| RMSE (Log Scale) | **0.0340** | 0.0404 |
| Within ±10% of True Price | **98.5%** | 97.8% |
| Within ±20% of True Price | **99.5%** | 99.3% |

**The tabular-only model outperformed the multimodal model on every metric.** Adding 50 PCA dimensions from CLIP introduced noise that slightly degraded accuracy.

This result makes sense in hindsight: wholesale tile pricing is driven almost entirely by physical specifications (size, material grade, certification tier) and product-line positioning (collection). Two tiles can look identical in their photos — same marble pattern, same colour — and differ in price because one is 6mm thick porcelain and the other is 20mm outdoor stoneware. The structured data already captures the pricing logic; the photos are largely redundant.

**The final submission uses the tabular-only model.** The notebook `03_final_submission.ipynb` runs both models, confirms this result, and automatically selects the winner.

---

## What Worked

- **`collection_mean_price`** was the single most important feature by a wide margin. The collection is a product-line quality tier — knowing which family a tile belongs to immediately narrows the price range.
- **`tile_volume_mm3`** was the strongest "raw" numeric predictor. Volume captures what area alone misses: thickness drives manufacturing cost, shipping weight, and breakage risk.
- **Finish keyword extraction** from product names outperformed the structured `finish_type` field. The structured field has 3 values; regex extraction yields 5+ meaningful categories.
- **Missingness flags** consistently ranked in the top 15 SHAP features. The absence of a certification result is itself a strong signal about product tier.
- **Log-transforming the target** was essential. Without it, the model over-optimises for the expensive tail and under-fits the dense $8–$15 cluster.

## What Didn't Work

- **CLIP image embeddings degraded accuracy.** The visual signal is real (marble tiles look different from concrete tiles), but the structured features already encode the pricing-relevant aspects. The 50 PCA dimensions added noise without adding predictive power. I kept the experiment in the code because it's an honest finding — worth showing even though the answer was "no."
- **Mean imputation** for missing technical ratings performed worse than treating NaN as a sentinel. XGBoost's native missing-value handling was superior to any imputation strategy.
- **Direct label encoding of `collection_name`** was useless. 200+ arbitrary integers carry no ordinal meaning. Target encoding (mean price per collection) was the right approach.
- **`color_family`** showed heavy overlap across price tiers and added negligible signal beyond what `finish_keyword` and `body_type` already captured.

---

## What I'd Do With More Time

1. **Fine-tune CLIP on tile images.** The current model uses a general-purpose encoder trained on internet photos. Fine-tuning on our 2,840 tile photos — using contrastive learning to pull same-collection tiles closer in embedding space — would produce far more tile-specific visual features and might actually improve on the tabular baseline.

2. **Neural collection embeddings.** Instead of hand-crafted mean/std features, learn a low-dimensional embedding for each collection jointly with price prediction, capturing richer within-collection structure.

3. **Extended hyperparameter search.** The current XGBoost parameters are reasonable defaults. A systematic Optuna search over 200+ trials across learning rate, depth, regularisation, and subsampling could squeeze out incremental gains.

4. **Model stacking.** XGBoost, LightGBM, and Ridge regression produce different error profiles. A meta-learner combining all three typically improves robustness, though at the cost of interpretability.

5. **Semi-supervised learning.** The 568 test-set images are available even though prices are hidden. These could improve PCA or CLIP fine-tuning without introducing label leakage.
