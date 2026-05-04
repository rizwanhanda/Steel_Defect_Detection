import pandas as pd
import os
import json

# Settings
CSV_PATH = 'train.csv'
IMAGE_DIR = 'test_samples'
OUTPUT_JSON = 'demo_truth.json'

def prepare_manifest():
    if not os.path.exists(CSV_PATH):
        print("Error: train.csv not found.")
        return

    # 1. Get the names of the 35 images you actually have
    demo_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
    
    # 2. Load CSV
    df = pd.read_csv(CSV_PATH)
    
    # 3. Adaptive Filtering (Handles both combined and split CSV formats)
    manifest = {}
    
    for filename in demo_files:
        manifest[filename] = {}
        for class_id in range(1, 5):
            # Try combined format first
            if 'ImageId_ClassId' in df.columns:
                target = f"{filename}_{class_id}"
                rle = df.loc[df['ImageId_ClassId'] == target, 'EncodedPixels']
            # Fallback to split format
            else:
                rle = df.loc[(df['ImageId'] == filename) & 
                            (df['ClassId'].astype(str) == str(class_id)), 'EncodedPixels']
            
            if not rle.empty and pd.notna(rle.iloc[0]):
                manifest[filename][str(class_id)] = rle.iloc[0]

    # 4. Save as a lightweight JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(manifest, f)
    
    print(f"✅ Success! Created {OUTPUT_JSON} with data for {len(demo_files)} images.")

if __name__ == "__main__":
    prepare_manifest()