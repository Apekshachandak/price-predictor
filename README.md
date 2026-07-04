# Tile Price Prediction — Write-Up



## How I Approached This

Before writing any code, I spent time brainstorming and stress-testing my plan. I structured this as a three-phase process — each phase documented in the `brainstorm/` folder:

1. **Grill Me** (`brainstorm/01_grill_me.md`) — I interrogated every assumption: why would a model work here, what's the biggest risk (collection leakage), why XGBoost over a neural net, why log-price, what if images don't help? The goal was to catch bad decisions before they cost time.

2. **Product Requirements** (`brainstorm/02_prd.md`) — I formalised the feature engineering plan: which fields to parse, what to engineer, how to handle missingness, how to use collection statistics without memorising them, and how to properly investigate image-fusion strategies.

3. **Issues Breakdown** (`brainstorm/03_prd_to_issues.md`) — I broke the work into concrete tasks with acceptance criteria, so I could track progress and avoid scope creep.

This upfront thinking shaped every downstream decision. Most of the "hard" work happened here — the code was just execution.

---

## What I Found in the Data

### Price Distribution
The 2,272 training tiles range from **$5.96 to $33.56/m²**, with a median around **$12.74/m²**. The distribution is right-skewed, and viewing it on a log scale reveals discrete pricing clusters at fixed intervals. This isn't a smooth market — it's structured manufacturer pricing, where tiles are placed into deliberate brackets tied to product tiers.

Training on `log(price)` and converting back at inference time normalises this skewness and ensures the model treats proportional errors fairly. Being off by $2 on a $10 tile is a bigger mistake than being off by $2 on a $30 tile — log-transformation encodes that intuition naturally.

### Collection Is the Dominant Signal
Every one of the 160 test collections is seen in training — zero unseen collections. Within-collection log-price standard deviation (median: 0.069) is less than a quarter of the overall spread (0.318). The collection is effectively a product-line quality tier, and knowing which family a tile belongs to immediately narrows the price range.

### Size Is the Strongest Numerical Lever
The README hinted that size drives per-m² price, and the data confirmed it emphatically. Each tile's dimensions are stored as nested JSON objects (`{"value": 1200.0, "comparator": "=", "unit": "mm"}`), which required careful parsing. From the extracted width, length, and thickness, I engineered:

- **Tile volume** (`width × length × thickness`) — Pearson correlation **r = 0.45** with log-price, the strongest single numeric feature. Bigger, thicker tiles cost more to manufacture, are heavier to ship, and break more easily in transit.
- **Tile area** (`width × length`) — r = 0.40. Volume beat area because the thickness dimension carries real signal: a 6mm slab-format tile and a 20mm outdoor paver have very different cost structures.
- **Aspect ratio** (`length / width`) — captures format differences. Long, narrow plank tiles (simulating wood) price differently from square tiles of the same area.

### Missingness Is Structural, Not Random
The dataset has 60+ fields, and many technical ratings (PEI wear, slip resistance, water absorption) are null for 30–65% of products. This isn't data quality noise — it's a structural signal. Budget tiles don't go through certification labs. When a tile has no PEI rating and no slip test result, it's almost certainly a basic wall tile, not a premium floor product.

Rather than imputing missing values (which would inject false information), I created boolean `_miss` flags for each technical field. XGBoost can then learn the rule: "if PEI rating is missing AND water absorption is missing, this tile is probably in the $8–$10 bracket." These flags turned out to be among the top features.

### Collection Encoding — Done Carefully
Tiles within the same collection share materials, finishes, and target markets. Their prices cluster tightly — most collections have a within-collection coefficient of variation below 0.05.

I used **empirical-Bayes (EB) shrinkage** for collection encoding: each collection is represented by its mean price, shrunk toward the global mean proportional to how many examples it has. A collection with only 2 SKUs shouldn't be trusted with a raw mean — shrinkage stabilises thin collections without affecting large, well-sampled ones. This is computed **exclusively from training data within each cross-validation fold** to prevent leakage.

`collection_mean_price` was the single most important feature in the model — confirming the intuition that product-line positioning explains most of the price variance.

### Finish Keywords from Product Names
The structured `finish_type` field only has three values (Matte, Glossy/Polished, and null). But the product name follows a consistent pattern — `[Collection] [Color] [Finish] [Size] [Extras]` — and encodes much richer finish information. A regex extraction yields keywords like `Pulido` (polished), `Brillo` (glossy), `Mate` (matte), `Lappato`, `Satin`, and `Antislip`, each with clean price separation.

### Categorical Features With Real Signal
A full audit of all 60+ fields surfaced several categorical features that meaningfully separate price tiers:

| Feature | Finding |
|---|---|
| **Body type** | Color-Body > Neutral-Body > White-Body > Red-Body. Clay composition directly affects material cost. |
| **Shade variation** | V3/V4 (high pattern variation) tiles are more expensive. Complex surface patterns require more manufacturing precision. |
| **Edge type** | Pressed/Cushioned edges vs. Rectified. Older decorative formats command premiums in niche segments. |
| **Barefoot slip rating** | Class C (highest grip) vs. Class A. High safety ratings indicate premium commercial-grade tiles. |
| **Application location** | Wall+Floor tiles (dual-purpose, more durable) command a premium over Wall-only tiles. |

---

## Features Engineered

| Feature | How It's Built | Why It Matters |
|---|---|---|
| `vol` | width × length × thickness | Strongest numeric predictor (r = 0.45); captures material mass |
| `area` | width × length | Direct manufacturing cost proxy |
| `aspect` | max(w,l) / min(w,l) | Plank vs. square format pricing |
| `finish_kw` | Regex from `product_name` | More granular than `finish_type` (Pulido/Brillo/Mate/Lappato/Satin/Antislip) |
| `app` | Parsed `application_location` list | Wall+Floor vs. Wall-only |
| `col_mean` | EB-shrunk mean log-price per collection (fold-safe) | Single most important feature |
| `col_std` | Std deviation per collection | Captures intra-collection variability |
| `col_cnt` | SKU count per collection | Feeds into shrinkage and is a proxy for line size |
| `*_miss` flags | 1 if field is null, 0 if present | Missingness correlates with product tier |
| `body_type`, `shade_variation_rating`, `edge_type` | Direct categoricals | Each separates price tiers clearly |
| `finish_type`, `color_family`, `subcategory`, `piece_type`, `is_glazed` | Direct categoricals | Additional market segment signals |

---

## Model and Training

### Why XGBoost (with LightGBM ensemble)
With 2,272 training rows, neural networks would overfit without heavy regularisation. XGBoost is built for exactly this regime: small-to-medium tabular data with missing values. It handles NaNs natively, trains in seconds, and produces interpretable predictions. For a submission where showing your reasoning matters as much as accuracy, interpretability is a feature, not a nice-to-have.

The final submission blends XGBoost + LightGBM with equal weight. Both models share the same CV structure but have different error profiles (different tree construction algorithms), so blending reduces variance.

### Cross-Validation Strategy
Two CV schemes were run:
- **Random 5-fold** — mirrors the real test distribution (all collections seen in training). This is the primary metric.
- **GroupKFold by collection** — the harder regime: some product lines are completely held out. Used to assess generalisation and understand where images add value.

Each fold computes collection statistics from **training rows only** — never touching the validation set. Empirical-Bayes shrinkage (`smooth=10.0`) stabilises small collections.

### Image Representations — Systematic Investigation
I ran two image representations and five fusion strategies — a proper ablation rather than one-shot test:

**Image Rep A — 22 compact "look" features (no DL):** Computed directly from pixel statistics — colour channels (HSV), colorfulness (Hasler-Süsstrunk), texture richness (Sobel edges, FFT high-frequency energy, image entropy), gloss (specular highlight area), tonal complexity. Human-interpretable, no GPU needed.

**Image Rep B — DINOv2 ViT-S/14 embeddings:** Self-supervised vision transformer, far stronger than CLIP for texture/material understanding. 384-dimensional embeddings extracted with multi-crop averaging (respecting the 4:1 tile strip format). Run on Kaggle GPU, PCA-reduced to 32 dimensions for the "direct" fusion variant.

**Five fusion strategies (all fold-safe):**
- `base` — tabular only
- `direct` — tabular + PCA-reduced image features concatenated
- `stack` — tabular + out-of-fold image→price prediction as a single meta-feature
- `knn` — tabular + mean log-price of the k=5 visually nearest training tiles (image kNN prior)
- `both` — stack + knn combined

---

## Results

Two CV regimes, 10+ experiments:

### Random 5-Fold CV (mirrors real test — all collections seen)
| Experiment | RMSE (log) | Within ±10% | Within ±20% |
|---|---|---|---|
| **tabular_EBshrink (WINNER)** | **0.0440** | **97.67%** | **98.94%** |
| ensemble_xgb_lgb | 0.0464 | 97.54% | 98.64% |
| dino_stack | 0.0471 | 97.45% | 98.64% |
| dino_base (tabular baseline) | 0.0448 | 97.40% | 98.86% |
| classical image (all fusions) | 0.0448–0.0492 | 97.0–97.4% | 98.5–98.9% |

### GroupKFold (unseen product lines — robustness regime)
| Experiment | RMSE (log) | Within ±10% |
|---|---|---|
| **dino_direct (BEST here)** | **0.1281** | **69.89%** |
| classical_both | 0.1358 | 66.37% |
| dino_both | 0.1351 | 66.11% |
| tabular_base | 0.1372 | 65.36% |

**The tabular model with EB-shrinkage is the final submission.** Adding image features doesn't improve the in-distribution metric — the collection encoding already saturates the price signal. However, DINOv2 gives a clear +4.5 percentage point lift under GroupKFold, proving images do carry real information for tiles outside known collections.

---

## What Worked

- **`col_mean` (EB-shrunk)** was the single most important feature. Shrinkage over raw mean is the right choice — it stabilises small collections without affecting large ones.
- **`vol` (tile volume)** was the strongest raw numeric predictor. Volume captures what area alone misses: thickness drives manufacturing cost, shipping weight, and breakage risk.
- **Finish keyword extraction** from product names outperformed the structured `finish_type` field. The structured field has 3 values; regex extraction yields 9 meaningful categories.
- **Missingness flags** consistently ranked in the top features. The absence of a certification result is itself a strong signal about product tier.
- **Log-transforming the target** was essential. Without it, the model over-optimises for the expensive tail and under-fits the dense $8–$15 cluster.
- **Two CV schemes** revealed where images add value — not on the easy in-collection task, but on the harder generalisation task.

## What Didn't Work

- **Image features didn't improve the in-distribution metric.** Five fusion strategies, two image representations — none moved the random CV metric. The structured features already encode the pricing logic; the photos are largely redundant when the collection is known.
- **DINOv2 does help under GroupKFold**, but that's the harder, less common evaluation regime. Direct concatenation (PCA-reduced DINOv2 embeddings) gave the best result there.
- **High-dimensional direct concatenation** of raw embeddings hurt more than it helped — the `direct` strategy was inferior to `base` on random CV. PCA reduction + noise introduced by irrelevant visual dimensions outweighed any signal.
- **kNN price priors from images** added noise under distribution shift — when validation collections weren't seen in training, the kNN neighbours were less reliable.

---

## What I'd Do With More Time

1. **Contrastively fine-tune DINOv2 on tile images.** The current model uses a general-purpose encoder trained on internet photos. Fine-tuning with a loss that pulls same-collection tiles closer in embedding space would produce tile-specific visual features and might genuinely lift the GroupKFold metric.

2. **Neural collection embeddings.** Instead of hand-crafted mean/std features, learn a low-dimensional embedding for each collection jointly with price prediction, capturing richer within-collection structure.

3. **Extended hyperparameter search.** The current XGBoost/LightGBM parameters are reasonable defaults. A systematic Optuna search over 200+ trials across learning rate, depth, regularisation, and subsampling could squeeze out incremental gains.

4. **Quantile models.** Instead of a point estimate, train models for the 10th and 90th percentile to produce calibrated price bands — more useful to a buyer than a single number.

5. **Semi-supervised learning.** The 568 test-set images are available even though prices are hidden. These could improve PCA or DINOv2 embeddings without introducing label leakage.
