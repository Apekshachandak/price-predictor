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
- The photo encodes things the structured data misses — marble veining, intricate patterns, colour richness. These aesthetics directly affect what a buyer is willing to pay.

So yes, I believe price is largely deterministic from these features — not random.

---

**Q: What's the biggest risk in your approach?**

A: Collection leakage. The collection average price is a very powerful feature — but if I compute it on the entire training set and use it as a feature, the model essentially memorises collections. I need to compute collection statistics in a fold-safe way during cross-validation (compute stats only from the training fold, not the validation fold). If I mess this up, my validation score will look great but the real test performance will be worse.

---

**Q: Why log(price) instead of raw price?**

A: The price distribution is right-skewed — most tiles cluster between $9–$20/m², with the full range running from $5.96 to $33.56/m². The log-scale plot also reveals discrete pricing clusters at fixed intervals, characteristic of structured manufacturer pricing brackets. If I train directly on raw price, the model spends too much effort optimising for the tail. Taking the logarithm compresses the scale into a near-bell distribution and ensures the model treats proportional errors fairly — being wrong by 20% at $10 gets the same weight as being wrong by 20% at $30.

---

**Q: Why XGBoost and not a neural network?**

A: A few reasons:
1. The dataset is small (~2,272 training rows). Neural networks typically need tens of thousands of rows to shine. XGBoost is specifically designed to work well on small-to-medium tabular data.
2. XGBoost handles missing values natively — it learns the best direction to send a missing value at each split. With heavy missingness in this dataset (many technical ratings are null), this matters a lot.
3. I can explain XGBoost predictions using SHAP. For an intern assignment, being able to say "tile area and collection tier drove this prediction" is more valuable than a black-box neural network that's slightly more accurate.
4. Speed — I can train and tune XGBoost on a laptop in minutes. A neural network would need more experimentation time.

---

**Q: Why are you bothering with the photos at all? Is it worth the extra complexity?**

A: The README explicitly says "Using it is optional but rewarded." More importantly, there's real signal there. Two tiles might have identical size, finish type listed as "Matte", and the same collection — but one has a subtle marble pattern and another is plain beige. The structured data would treat them identically. The photo reveals the difference. I'm not training a custom image model — I'm just running a pre-trained one (CLIP) on the photos to get feature vectors, which only takes ~10 minutes on a free GPU. The marginal effort is low; the potential gain is real.

---

**Q: What if your CLIP image embeddings don't help? What's your fallback?**

A: The structured-feature-only XGBoost model is already a complete, standalone submission. The image embeddings are an additive experiment. If they don't improve the validation score, I'll mention it in the write-up: "I tried adding visual features extracted using OpenAI's CLIP model. The cross-validation score did not improve, suggesting that the structured attributes already capture most of the price-relevant information in this dataset." That's a valid, honest finding — not a failure.

---

**Q: How will you handle the ~200 collection names? There are too many to one-hot encode.**

A: I won't one-hot encode them. I'll use two strategies:
1. Target encoding — replace each collection name with the mean log-price of that collection, computed in a fold-safe way.
2. Aggregated statistics — collection mean price, standard deviation of prices, count of products in the collection. These give the model both the average tier and the variability within a collection.

For test set collections that don't appear in training at all, I'll fall back to the global mean price. There should be very few (or zero) such cases since the collections are shared between train and test.

---

**Q: How will you evaluate your model honestly before submitting?**

A: 5-fold cross-validation on the training set. I split the 2,272 training rows into 5 equal parts, train on 4 parts, validate on the 5th, and rotate. I report:
- Mean Absolute Error on actual price (interpretable: "on average off by $X")
- Root Mean Squared Error on log-price (the standard regression metric)
- Percentage of predictions within ±10% of true price (what the README asks for)

The out-of-fold predictions give me an honest estimate of how the model will perform on unseen data without ever touching the test set.

---

**Q: What's the one thing you'd do differently if you had 2 more weeks?**

A: I'd fine-tune CLIP specifically on tile images. Right now I'm using a general-purpose model trained on internet photos — it's never seen ceramic tiles. Fine-tuning it on our 2,840 tile photos with price as a supervision signal (or better yet, using contrastive learning between similar-collection tiles) could extract much richer visual features. That would likely be the biggest accuracy gain available at this point.
