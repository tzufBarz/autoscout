import cv2
from pathlib import Path
import csv

crops_dir = Path("output/images")
output_csv = Path("output/labels.csv")

image_files = sorted(crops_dir.glob("*.jpg"))

# find already labeled images to resume if interrupted
labeled = set()
if output_csv.exists():
    with open(output_csv) as f:
        for row in csv.reader(f):
            labeled.add(row[0])

with open(output_csv, "a", newline="") as f:
    writer = csv.writer(f)
    
    for image_path in image_files:
        if str(image_path) in labeled:
            continue
            
        img = cv2.imread(str(image_path))
        
        if img is None:
            continue
        
        # scale up for visibility if crop is small
        h, w = img.shape[:2]
        scale = max(1, 300 // h)
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("label", img)
        cv2.waitKey(30)
        
        label = input(f"{image_path.name}: ").strip()
        
        if label == "s":  # skip unclear images
        	continue
        if label == "q":  # quit and save progress
            break
            
        writer.writerow([str(image_path), label])
        f.flush()

cv2.destroyAllWindows()