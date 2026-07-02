"""
KAGGLE GPU SCRIPT — CLIP Image Embedding Extraction
====================================================
Run this on Kaggle with GPU enabled (Settings → Accelerator → GPU T4 x2).

Steps to run:
1. Go to kaggle.com → Create New Notebook
2. Settings → Accelerator → GPU T4 x2
3. Upload your images/ folder as a Kaggle Dataset
4. Paste this entire script and run it
5. Download clip_embeddings.npy and clip_sku_ids.npy
6. Put them in your local tile_pricing_challenge/embeddings/ folder
"""

# Install the CLIP library
import subprocess
subprocess.run(["pip", "install", "open-clip-torch", "-q"])

import open_clip
import torch
from PIL import Image
import numpy as np
import os

print("Step 1: Loading CLIP model (ViT-B/32 with OpenAI weights)...")
# ViT-B/32 = Vision Transformer, Base size, 32x32 patch size
# This model was pre-trained by OpenAI on 400 million image-text pairs
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)
model.eval()  # Evaluation mode: no dropout, no weight updates
print("Model loaded.")

# ----------------------------------------------------------------
# CHANGE THIS PATH to wherever you uploaded your images/ folder
# ----------------------------------------------------------------
IMAGES_DIR = "/kaggle/input/tile-pricing-challenge/images/"
# ----------------------------------------------------------------

print(f"\nStep 2: Scanning images in {IMAGES_DIR}")
image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
print(f"Found {len(image_files)} images.")

print("\nStep 3: Extracting embeddings...")
all_embeddings = {}
BATCH_SIZE = 64  # Process 64 images at a time for efficiency

with torch.no_grad():  # Don't store gradients — saves GPU memory
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i + BATCH_SIZE]
        batch_images = []
        batch_ids = []

        for filename in batch_files:
            sku_id = filename.replace(".jpg", "")
            img_path = os.path.join(IMAGES_DIR, filename)
            try:
                img = preprocess(Image.open(img_path).convert("RGB"))
                batch_images.append(img)
                batch_ids.append(sku_id)
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                # Use zero vector as fallback for broken images
                all_embeddings[sku_id] = np.zeros(512)

        if batch_images:
            batch_tensor = torch.stack(batch_images)  # Shape: (batch, 3, 224, 224)
            embeddings = model.encode_image(batch_tensor)
            # Normalise each embedding to unit length (standard practice with CLIP)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings_np = embeddings.cpu().numpy()

            for sku_id, emb in zip(batch_ids, embeddings_np):
                all_embeddings[sku_id] = emb

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(image_files))}/{len(image_files)} images...")

print(f"\nStep 4: Saving embeddings...")
sku_ids = list(all_embeddings.keys())
embedding_matrix = np.array([all_embeddings[s] for s in sku_ids])

print(f"Embedding matrix shape: {embedding_matrix.shape}")  # Should be (2840, 512)

np.save("clip_embeddings.npy", embedding_matrix)
np.save("clip_sku_ids.npy", np.array(sku_ids))

print("\nDone! Download these two files from Kaggle:")
print("  - clip_embeddings.npy")
print("  - clip_sku_ids.npy")
print("\nPlace them in your local embeddings/ folder and run 02_model.ipynb Step 10.")
