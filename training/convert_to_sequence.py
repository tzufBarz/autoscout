import os
from pathlib import Path

def convert_yolo_to_sequence(label_path):
    digits = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            cx = float(parts[1])
            digits.append((cx, cls))
    
    digits.sort(key=lambda d: d[0])
    return "".join(str(d[1]) for d in digits)

labels_dir = Path("labels")
output = []
for label_file in labels_dir.glob("*.txt"):
    image_file = label_file.with_suffix(".jpg")
    sequence = convert_yolo_to_sequence(label_file)
    if sequence:
        output.append(f"{image_file},{sequence}")

with open("dataset.csv", "w") as f:
    f.write("\n".join(output))