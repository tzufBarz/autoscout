import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import subprocess

def ensure_h264(video_path: Path) -> Path:
    out_path = video_path.with_suffix(".h264.mp4")
    if not out_path.exists():
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-c:v", "libx264", "-crf", "23",
            "-y", str(out_path)
        ], check=True, capture_output=True)
    return out_path


def nms_boxes(boxes, iou_thresh=0.5):
    keep = []
    for i, b1 in enumerate(boxes):
        duplicate = False
        for b2 in keep:
            # compute IoU
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            iou = inter / (area1 + area2 - inter)
            if iou > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            keep.append(b1)
    return keep


def perceptual_hash(crop):
    small = cv2.resize(crop, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    return (gray > mean).flatten()  # 256-bit binary hash

def too_similar(hash1, hash2, thresh=0.1):
    # fraction of differing bits
    return (hash1 != hash2).mean() < thresh

saved_hashes = []


VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
ROBOT_MODEL_PATH = "../backend/models/robots.pt"
DIGIT_MODEL_PATH = "../backend/models/digits.pt"
FRAMES_PER_VIDEO = 100
ROBOT_CONF_THRESH = 0.25
DIGIT_CONF_THRESH = 0.3

robot_model = YOLO(ROBOT_MODEL_PATH)
# digit_model = YOLO(DIGIT_MODEL_PATH)

output_images = Path(OUTPUT_DIR) / "images"
# output_labels = Path(OUTPUT_DIR) / "labels"
output_images.mkdir(parents=True, exist_ok=True)
# output_labels.mkdir(parents=True, exist_ok=True)

video_dir = Path(VIDEO_DIR)

for video_path in video_dir.glob("*.mp4"):
    if "h264" in video_path.name:
        continue
    video_path = ensure_h264(video_path)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, FRAMES_PER_VIDEO, dtype=int)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        robot_results = robot_model(frame, conf=ROBOT_CONF_THRESH, verbose=False)[0]
        if robot_results.boxes is None:
            continue

        boxes = robot_results.boxes.xyxy.cpu().numpy()
        boxes = nms_boxes(boxes)
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            h = perceptual_hash(crop)
            if not any(too_similar(h, existing) for existing in saved_hashes):
                saved_hashes.append(h)
                stem = f"{video_path.stem}_f{frame_idx}_r{i}"
                image_path = output_images / f"{stem}.jpg"
                # label_path = output_labels / f"{stem}.txt"

                cv2.imwrite(str(image_path), crop)

            # digit_results = digit_model(crop, conf=DIGIT_CONF_THRESH, verbose=False)[0]

            # with open(label_path, "w") as f:
            #     if digit_results.boxes is not None:
            #         for j in range(len(digit_results.boxes)):
            #             dbox = digit_results.boxes.xywh[j].cpu().numpy()
            #             dcls = digit_results.boxes.cls[j].cpu().numpy()
            #             cx, cy, w, h = dbox
            #             crop_h, crop_w = crop.shape[:2]
            #             f.write(f"{int(dcls)} {cx/crop_w:.6f} {cy/crop_h:.6f} {w/crop_w:.6f} {h/crop_h:.6f}\n")
    
    cap.release()
    print(f"Done: {video_path.name}")

print("All videos processed.")