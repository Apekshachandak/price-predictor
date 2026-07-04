# Grill Me — Stress-Testing My Approach

> Before I wrote a single line of code, I stress-tested my plan by grilling myself on every decision.
> The goal was to make sure I wasn't just guessing — that every choice had a reason.

---

**Q: What exactly is the problem you're solving?**

A: A tile manufacturer has 2,840 products. Each product has a bunch of attributes — size, colour, finish, technical quality ratings — and a photo of the tile surface. For 2,272 of them, I know the wholesale price in USD per square metre. I need to predict the price for the remaining 568 products.

---

**Q: Why would a model even work here? What makes you think price can be predicted from these attributes?**

A: A few strong intuitions backed by domain knowledge:
- Larger tiles cost more to manufacture and are harder to handle, so size is directly linked to price.
- The collection a tile belongs to is basically a brand/quality tier proxy. Tiles in the same collection share materials, finishes, and target markets — their prices cluster together.
- Technical ratings like wear resistance (PEI rating) and water absorption signal quality tier. A tile rated for heavy commercial use costs more than one rated for light residential.
- The photo encodes things the structured data misses — marble veining, intricate patterns, surface texture, gloss. These aesthetics directly affect what a buyer is willing to pay.

So yes, I believe price is largely deterministic from these features — not random.

---

**Q: What's the biggest risk in your approach?**

A: Collection leakage. The collection average price is a very powerful feature — but if I compute it on the entire training set and use it as a feature, the model essentially memorises collections. I need to compute collection statistics in a fold-safe way during cross-validation (compute stats only from the training fold, not the validation fold). If I mess this up, my validation score will look great but the real test performance will be worse.

A second subtle risk: small collections. A collection with only 2 or 3 SKUs will have a very noisy mean price. I address this with empirical-Bayes shrinkage — the smaller the collection, the more its mean is pulled toward the global average. This stabilises thin collections without affecting large, well-sampled ones.

---

**Q: Why log(price) instead of raw price?**

A: The price distribution is right-skewed — most tiles cluster between $9–$20/m², with the full range running from $5.96 to $33.56/m². The log-scale plot also reveals discrete pricing clusters at fixed intervals, characteristic of structured manufacturer pricing brackets. If I train directly on raw price, the model spends too much effort optimising for the tail. Taking the logarithm compresses the scale into a near-bell distribution and ensures the model treats proportional errors fairly — being wrong by 20% at $10 gets the same weight as being wrong by 20% at $30.

---

**Q: Why XGBoost and not a neural network?**

A: A few reasons:
1. The dataset is small (~2,272 training rows). Neural networks typically need tens of thousands of rows to shine. XGBoost is specifically designed to work well on small-to-medium tabular data.
2. XGBoost handles missing values natively — it learns the best direction to send a missing value at each split. With heavy missingness in this dataset (many technical ratings are null), this matters a lot.
3. Speed — I can train and tune XGBoost on a laptop in minutes. I also blend it with LightGBM for a slightly better error profile. Both models are complementary in how they construct trees, so a 50/50 blend gives more robust predictions than either alone.

---

**Q: Why are you bothering with the photos at all? Is it worth the extra complexity?**

A: The README explicitly says "Using it is optional but rewarded." More importantly, there's real signal there. Two tiles might have identical size, finish type listed as "Matte", and the same collection — but one has a subtle marble pattern and another is plain beige. The structured data would treat them identically. The photo reveals the difference.

I'm not training a custom image model — I'm running DINOv2 (a self-supervised Vision Transformer from Meta, pre-trained on curated images) frozen on the tile photos to get rich 384-dimensional visual embeddings. DINOv2 is particularly good at texture and material understanding, which is exactly what tile photos are about. The marginal effort is low — run it on Kaggle GPU, cache the embeddings, done.

---

**Q: What if the image features don't help the main metric? What's your fallback?**

A: I ran five fusion strategies systematically so I'd have a real answer, not a guess. The finding was nuanced: under random 5-fold CV (which mirrors the real test — all collections seen in training), no image strategy improved on the tabular-only model. The collection encoding already saturates the metric. But under GroupKFold (where some product lines are completely held out of training), DINOv2 features gave a meaningful boost. So the answer isn't "images don't help" — it's "images don't help when the collection is known, but they matter for generalisation." I document this honestly. The final submission uses the tabular model because the real test is in-collection; the image experiment is a legitimate finding, not a failure.

---

**Q: How will you handle the ~200 collection names? There are too many to one-hot encode.**

A: I use empirical-Bayes target encoding:
- Each collection is represented by its shrunk mean log-price (`smooth=10`), standard deviation, and count — computed from training rows only, inside each CV fold.
- The shrinkage parameter pulls small collections toward the global mean, preventing overfit on collections with only 1–3 SKUs.
- For test collections not in training (zero in this dataset, but handled as a fallback), I use the global mean.

This gives the model both the average price tier and the variability within a collection — far more informative than a raw integer label or one-hot columns.

---

**Q: You mentioned two CV schemes. Why both?**

A: Random 5-fold mirrors the actual test condition — every test collection appears in training, so the model just needs to generalise within product lines. This is the primary metric and the metric that drives the final submission.

GroupKFold (splitting by collection, so some product lines are completely unseen at validation time) is a harder, more realistic scenario. It tells me how the model would behave if the manufacturer added a brand-new collection. It's also the regime where images are most valuable. Running both gave me a complete picture of model behaviour, not just the leaderboard number.

---

**Q: How will you evaluate your model honestly before submitting?**

A: Out-of-fold predictions from 5-fold cross-validation. I split the 2,272 training rows into 5 equal parts, train on 4 parts, validate on the 5th, and rotate. I report:
- Root Mean Squared Error on log-price (the standard regression metric)
- Percentage of predictions within ±10% of true price (what the README asks for)
- Percentage of predictions within ±20% of true price (robustness check)

The final submission refits on all 2,272 rows — blending XGBoost and LightGBM with EB-shrunk collection encoding — and generates predictions for all 568 test SKUs.

---

**Q: What's the one thing you'd do differently if you had 2 more weeks?**

A: I'd fine-tune DINOv2 specifically on tile images using contrastive learning — pulling tiles from the same collection closer in embedding space. Right now I'm using frozen embeddings from a general-purpose model. Tile-specific fine-tuning would make the visual features far more discriminative for price prediction, especially in the GroupKFold (unseen collection) regime where images already show promise. That would likely be the biggest accuracy gain available at this point.
