import cv2
from fastapi import FastAPI, UploadFile, Form
import uuid, asyncio, shutil
import os

from ultralytics import YOLO
import numpy as np
import Levenshtein
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict


ALLIANCE_WEIGHT = 0.5
DIGIT_WEIGHT = 1 - ALLIANCE_WEIGHT

ALPHA = 0.3
BETA = 1 - ALPHA

TEAM_THRESH = 0.75

ALLIANCE_MASK = (np.arange(6) // 3)


schedule = {
    1: [
        "5951", "3339", "4320", # Blue
        "5654", "1690", "5990"   # Red
    ]
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(BASE_DIR, "models/robots.pt"))
digit_model = YOLO(os.path.join(BASE_DIR, "models/digits.pt"))


app = FastAPI()

jobs = {}

@app.post("/upload")
async def upload(file: UploadFile, match: int = Form()):
    job_id = str(uuid.uuid4())
    path = f"/tmp/{job_id}.mp4"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"status": "processing", "result": None, "progress": None}
    asyncio.create_task(process(job_id, path, match))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    return jobs.get(job_id, {"status": "not_found"})

async def process(job_id: str, path: str, match: int):
    try:
        def progress_callback(info: dict):
            jobs[job_id]["progress"] = info

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_pipeline, path, match, progress_callback)
        jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "result": str(e)}


def detect_digits(crops: list, digit_model, conf_thresh=0.4) -> list[str | None]:
    if not crops:
        return []

    batch_results = digit_model(crops, conf=conf_thresh, verbose=False)

    output = []
    for results in batch_results:
        if results.boxes is None:
            output.append(None)
            continue

        boxes = results.boxes.xywh.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        digits = [(box[0], cls, score) for box, score, cls in zip(boxes, scores, classes)]

        if not digits:
            output.append(None)
            continue

        output.append("".join(str(d[1]) for d in sorted(digits, key=lambda d: d[0])))
    
    return output


def digit_scores(digits: str, teams: list[str]) -> np.ndarray:
    scores = np.zeros(6)
    for i, team in enumerate(teams):
        dist = Levenshtein.distance(digits, team)
        max_len = max(len(digits), len(team))
        score = 1 - (dist / max_len)
        scores[i] = score
    return scores


def update(track: int, teams: list[str], alliance: int, digits: str, track_votes: dict, track_teams: dict, team_tracks) -> None:
    votes = (digit_scores(digits, teams) * DIGIT_WEIGHT) + \
        ((ALLIANCE_MASK == alliance) * ALLIANCE_WEIGHT)
    if track not in track_votes:
        track_votes[track] = np.zeros(6)
    track_votes[track] = ALPHA * track_votes[track] + BETA * votes
    match = np.argmax(track_votes[track] * (team_tracks[np.arange(6)] == 0))
    if track_votes[track][match] >= TEAM_THRESH:
        track_teams[track] = match
        team_tracks[match] = track


def smooth_and_interpolate(positions, sigma=5):
    if not positions:
        return {}

    # Step 1: average duplicate/close frames (in case a track has multiple hits per frame)
    frame_buckets = defaultdict(list)
    for f, x, y in positions:
        frame_buckets[f].append((x, y))
    
    known = {f: np.mean(pts, axis=0) for f, pts in frame_buckets.items()}
    known_frames = sorted(known.keys())

    # Step 2: interpolate across all frames between first and last known point
    all_frames = np.arange(known_frames[0], known_frames[-1] + 1)
    xs = np.interp(all_frames, known_frames, [known[f][0] for f in known_frames])
    ys = np.interp(all_frames, known_frames, [known[f][1] for f in known_frames])

    # Step 3: gaussian smooth the interpolated signal
    xs = gaussian_filter1d(xs, sigma=sigma)
    ys = gaussian_filter1d(ys, sigma=sigma)

    return dict(zip(all_frames, zip(xs, ys)))


def run_pipeline(video_path: str, match_number: int, progress_callback=None) -> dict:
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    teams = schedule[match_number]

    results = model.track(
        source=video_path,
        tracker=os.path.join(BASE_DIR, "frc_botsort.yaml"),
        persist=True,
        stream=True,
        verbose=False,
        conf=0.1
    )

    track_votes = {}
    track_teams = {}
    team_tracks = np.zeros(6, dtype=int)


    track_positions = {}

    with tqdm(total=frame_count) as pbar:
        for frame_idx, result in tqdm(enumerate(results)):
            if not result.boxes or result.boxes.xyxy is None or result.boxes.id is None or result.boxes.cls is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            tracker = model.predictor.trackers[0]
            alive_ids = set(t.track_id for t in tracker.tracked_stracks)
            lost_ids = set(t.track_id for t in tracker.lost_stracks)

            for i, track in enumerate(team_tracks):
                if track != 0 and track not in alive_ids and track not in lost_ids:
                    team_tracks[i] = 0

            crops = []
            crop_tracks = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                tid = int(ids[i])
                alliance = int(classes[i])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if tid not in track_positions:
                    track_positions[tid] = []
                track_positions[tid].append((frame_idx, cx, cy))
                
                if not tid in track_teams and y2 > y1 and x2 > x1:
                    crops.append(result.orig_img[y1:y2, x1:x2])
                    crop_tracks.append((tid, alliance))


            numbers = detect_digits(crops, digit_model)
            for track, digits in zip(crop_tracks, numbers):
                if digits:
                    tid, alliance = track
                    update(tid, teams, alliance, digits, track_votes, track_teams, team_tracks)

            pbar.update()
            if progress_callback:
                progress_callback({
                    "frame": frame_idx,
                    "total": frame_count,
                    "percent": round(frame_idx / frame_count * 100, 1),
                    "it_per_s": round(pbar.format_dict.get("rate") or 0, 2),
                    "elapsed": round(pbar.format_dict.get("elapsed") or 0, 1),
                    "eta": round((frame_count - frame_idx) / pbar.format_dict["rate"], 1)
                        if pbar.format_dict.get("rate") else None,
                })


    all_frame_positions = []
    for team_idx in range(len(teams)):
        tids = [tid for tid, idx in track_teams.items() if idx == team_idx]
        positions = []
        for tid in tids:
            positions.extend(track_positions.get(tid, []))
        positions.sort(key=lambda p: p[0])
        all_frame_positions.append(smooth_and_interpolate(positions))

    return {
        "teams": teams,
        "frame_count": frame_count,
        "fps": frame_rate,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "trajectories": {
            teams[i]: [
                {"frame": int(f), "x": float(x), "y": float(y)}
                for f, (x, y) in frame_pos.items()
            ]
            for i, frame_pos in enumerate(all_frame_positions)
        }
    }
