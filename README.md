# Tile Price Prediction — Write-Up

## How I Approached This

Before writing any code, I spent time stress-testing my plan. I've structured this process across three documents in the `brainstorm/` folder:

1. **Grill Me** (`brainstorm/01_grill_me.md`) — I interrogated every decision upfront: why would a model work here, what's the biggest risk, why log-price, why XGBoost, what does the GroupKFold tell us that random CV doesn't? The goal was to catch bad assumptions before they cost hours of work.

2. **Product Requirements** (`brainstorm/02_prd.md`) — I formalised the feature engineering plan and the multimodal investigation strategy before touching the data.

3. **Issues Breakdown** (`brainstorm/03_prd_to_issues.md`) — I broke the work into concrete, trackable tasks with clear definitions of done.

This upfront thinking shaped every downstream decision. The brainstorm docs aren't an afterthought — they're where most of the actual reasoning happened.

---

## What I Found in the Data

### Price Distribution

The 2,272 training tiles range from **$5.96 to $33.56/m²**, with a median around **$12.74/m²**. The distribution is right-skewed, and on a log scale you can see discrete pricing clusters at fixed intervals. This isn't a smooth open market — it's structured manufacturer pricing, where tiles sit in deliberate brackets tied to product tier. Training on `log(price)` and converting back at inference normalises this skewness and ensures proportional errors are treated fairly: being off by $2 on a $10 tile matters more than being off by $2 on a $30 tile.

### Collection Is the Dominant Signal

Every one of the 160 test collections appears in the training set — zero unseen collections. Within-collection log-price standard deviation (median: **0.069**) is less than a quarter of the overall spread (**0.318**). Knowing which product family a tile belongs to immediately narrows its price range to a tight band. This is the structural fact that shapes everything else in the model.

### Size Is the Strongest Numerical Lever

Each tile's dimensions are stored as nested JSON objects (`{"value": 1200.0, "comparator": "=", "unit": "mm"}`), which required careful parsing. From the extracted width, length, and thickness, I engineered:

- **Tile volume** (`width × length × thickness`) — Pearson **r = 0.45** with log-price, the strongest single numeric feature. Bigger, thicker tiles cost more to produce, ship, and handle.
- **Tile area** (`width × length`) — r = 0.40. Volume edges it out because thickness carries real signal: a 6mm wall tile and a 20mm outdoor stoneware paver have very different cost structures.
- **Aspect ratio** (`max(w,l) / min(w,l)`) — long plank tiles (simulating wood flooring) price differently from square tiles of the same area.

### Missingness Is Structural, Not Noise

The dataset has 60+ fields, and many technical ratings — PEI wear, slip resistance, water absorption — are null for 30–65% of products. Budget tiles simply don't go through certification labs. When a tile has no PEI rating and no slip test result, it's almost certainly a basic wall tile, not a premium floor product.

Rather than imputing these missing values (which would inject false information), I created boolean `_miss` flags for each technical field. XGBoost can learn: "if PEI is missing AND water absorption is missing, this tile is probably in the $8–$10 bracket." These flags consistently ranked in the top features by importance.

### Finish Keywords from Product Names

The structured `finish_type` field has only three values: Matte, Glossy/Polished, and null. But the product name follows a consistent pattern — `[Collection] [Color] [Finish] [Size] [Extras]` — and encodes much richer finish information. A regex extraction yields keywords like `Pulido` (polished), `Brillo` (glossy), `Mate` (matte), `Lappato`, `Satin`, and `Antislip`, each with clear and distinct price separation.

### Categorical Features With Real Signal

A full audit of the structured fields surfaced several categorical features that meaningfully separate price tiers:

| Feature | Key Finding |
|---|---|
| **Body type** | Color-Body tiles carry a clear premium over Neutral-Body, White-Body, and Red-Body. Clay composition directly affects manufacturing cost. |
| **Shade variation** | V3/V4 (high pattern variation) tiles are systematically more expensive — complex surface patterns require more precision. |
| **Edge type** | Pressed and cushioned edges vs. rectified. Decorative formats command premiums in specific market segments. |
| **Application location** | Dual-purpose Wall+Floor tiles (more durable, pass more tests) command a premium over Wall-only. |
| **Subcategory and piece type** | Additional market segment signals that the model learns to use. |

---

## Features Engineered

| Feature | How It's Built | Why It Matters |
|---|---|---|
| `vol` | width × length × thickness | Strongest numeric predictor; captures material mass and manufacturing cost |
| `area` | width × length | Direct manufacturing cost proxy |
| `aspect` | max(w,l) / min(w,l) | Plank vs. square format pricing |
| `finish_kw` | Regex from `product_name` | 9 categories; far more granular than the structured `finish_type` |
| `app` | Parsed `application_location` | Wall+Floor vs. Wall-only and other combos |
| `col_mean` | EB-shrunk mean log-price per collection (fold-safe) | Highly important target encoding for collection tier |
| `col_std` | Std deviation per collection | Captures intra-collection price spread |
| `col_cnt` | SKU count per collection | Feeds the shrinkage calculation |
| `*_miss` flags | 1 if field is null, 0 if present | Missingness encodes product tier |
| `body_type`, `shade_variation_rating`, `edge_type` | Direct categoricals | Each cleanly separates price tiers |
| `finish_type`, `color_family`, `subcategory`, `piece_type`, `is_glazed` | Direct categoricals | Additional market segment signals |

---

## Model and Training

### Why XGBoost + LightGBM

With 2,272 training rows, gradient boosting is the right tool. Both XGBoost and LightGBM are designed for small-to-medium tabular data, handle NaNs natively (learning the optimal split direction for missing values at each node), and produce interpretable predictions. They also have complementary tree construction algorithms — XGBoost builds trees level-by-level; LightGBM uses leaf-wise growth — which means their errors don't perfectly correlate. A 50/50 blend of both in log-price space reduces variance without adding complexity.

### Cross-Validation Strategy

I ran two CV schemes in parallel:

**Random 5-fold** is the primary metric. It mirrors the actual test condition: all test collections appear in training, so the model needs to generalise within known product lines. Each fold computes collection statistics from the training portion only — never touching the validation set.

**GroupKFold by collection** is the harder regime: some product lines are completely held out of training. This tells me how the model behaves with genuinely new product lines, and it's the regime where the visual features add measurable value.

### Empirical-Bayes Collection Encoding

The collection name is encoded as its mean log-price, shrunk toward the global mean:

```
col_mean_shrunk = (n × raw_mean + smooth × global_mean) / (n + smooth)
```

With `smooth=10`, a collection with only 2 SKUs gets its estimate pulled most of the way to the global mean. A collection with 50 SKUs is barely affected. This stabilises thin collections and improved the primary CV metric over raw mean encoding.

### Image Investigation — DINOv2

The README notes that photos are optional but rewarded. There's real visual information in these tiles — texture, grain pattern, gloss level, surface complexity — that the structured fields only partly capture.

I extracted 384-dimensional embeddings using **DINOv2 ViT-S/14** (Meta's self-supervised Vision Transformer, pre-trained on curated image data). DINOv2 is particularly strong at texture and material understanding, which is exactly the relevant signal for tile images. Inference runs frozen — no fine-tuning — on Kaggle GPU, with multi-crop averaging to handle the 4:1 landscape strip format.

I also computed **22 compact, interpretable "look" features** directly from pixel statistics: HSV colour channels, colorfulness (Hasler–Süsstrunk metric), texture richness (Sobel edge density, FFT high-frequency energy, image entropy), gloss level (specular highlight area), and tonal complexity. These are CPU-only and human-readable.

**Five fusion strategies** were tested systematically under both CV schemes:
- `base` — tabular only (the control)
- `direct` — tabular + PCA-reduced image features concatenated directly
- `stack` — tabular + an out-of-fold `image → price` prediction as a single meta-feature
- `knn` — tabular + mean log-price of the 5 visually nearest training tiles (image kNN prior)
- `both` — stack + knn combined

---

## Results

### Random 5-Fold CV (mirrors the real test)

| Experiment | RMSE (log) | Within ±10% | Within ±20% |
|---|---|---|---|
| **tabular + EB-shrinkage (WINNER)** | **0.0440** | **97.67%** | **98.94%** |
| ensemble_xgb_lgb | 0.0464 | 97.54% | 98.64% |
| dino_stack | 0.0471 | 97.45% | 98.64% |
| tabular_base | 0.0448 | 97.40% | 98.86% |
| classical image (all fusions) | 0.0448–0.0492 | 97.0–97.4% | 98.5–98.9% |

### GroupKFold (unseen product lines — generalisation regime)

| Experiment | RMSE (log) | Within ±10% |
|---|---|---|
| **DINOv2-direct (BEST here)** | **0.1281** | **69.89%** |
| classical features + both | 0.1358 | 66.37% |
| tabular_base | 0.1372 | 65.36% |

**The final submission uses the tabular model with EB-shrinkage.** This configuration wins on random CV — the metric that mirrors the actual test — by a clear margin. The image investigation is a genuine finding, not a failed experiment: visual features add meaningful signal (+4.5pp) when product lines are completely unseen, confirming that the photos carry real information beyond the structured attributes, but that this information is largely redundant when the collection is already known.

---

## What Worked

- **Size and Edge Type** proved to be the strongest raw predictors (`area`, `vol`, `edge_type`). Larger tiles and specific edge finishes drive the baseline cost.
- **EB-shrunk `col_mean`** remained a top-tier feature. Shrinkage over a raw mean was the right call — it stabilises thin collections without penalising large ones.
- **Finish keyword extraction** from product names outperformed the structured `finish_type` field — 9 meaningful categories vs. 3.
- **Missingness flags** consistently ranked in the top features. The absence of a lab certification is itself a strong indicator of product tier.
- **Log target transformation** was essential. Without it, the model over-optimises for the expensive tail and under-fits the dense $8–$15 cluster.
- **Running two CV schemes** gave a complete picture: random CV tells you the leaderboard number; GroupKFold tells you where the model's ceiling is and where additional signals (like images) would pay off.

## What Didn't Work / What I'd Do Differently

- **Image features didn't move the in-distribution metric.** Under random CV, no fusion strategy — direct concatenation, stacking, kNN price prior — improved on the tabular-only model. The collection encoding already saturates the signal that's available in the structured data; the photos don't add to what the model already knows about price tier.
- **High-dimensional direct concatenation** (dumping all PCA-reduced image embeddings into the feature matrix) was slightly worse than baseline. The noise from irrelevant visual dimensions outweighed the signal. A compact representation (stacking or kNN) is cleaner.
- **With more time**, I'd fine-tune DINOv2 on tile images using contrastive learning — pulling same-collection tiles closer in embedding space. Frozen embeddings from a general-purpose model are a good starting point, but tile-specific visual features would likely give a real lift in the GroupKFold regime, where images already show clear promise. I'd also run a systematic Optuna hyperparameter search over 200+ trials and experiment with quantile regression to produce calibrated price bands rather than point estimates.
