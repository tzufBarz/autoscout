from typing import Annotated

import cv2
from fastapi import FastAPI, File, UploadFile, Form, Path
import uuid, asyncio, shutil
from uuid import UUID
import os

from ultralytics import YOLO
import numpy as np
import Levenshtein
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict


# Constants
ALLIANCE_WEIGHT = 0.5
DIGIT_WEIGHT = 1 - ALLIANCE_WEIGHT

ALPHA = 0.3
BETA = 1 - ALPHA

TEAM_THRESH = 0.75

ALLIANCE_MASK = (np.arange(6) // 3)

SAMPLE_RATE = 50 # Hz


# Match schedule: the teams in each alliance (blue followed by red) for each match number
schedule = {
    1: [
        "5951", "3339", "4320",
        "5654", "1690", "5990",
    ]
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
model = YOLO(os.path.join(BASE_DIR, "models/robots.pt"))
digit_model = YOLO(os.path.join(BASE_DIR, "models/digits.pt"))

# FastAPI setup
app = FastAPI()

jobs = {}

@app.post("/upload")
async def upload(file: Annotated[UploadFile, File(description="The video file")], match: Annotated[int, Form(description="The match number")]):
    """
    Upload a new video and begin tracking.
    Args:
        file: The video file.
        match: The match number.
    Returns:
        UUID: The ID for this job.
    """
    job_id = uuid.uuid4()
    path = f"/tmp/{job_id}.mp4"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    jobs[job_id] = {"status": "processing", "result": None, "progress": None}
    asyncio.create_task(process(job_id, path, match))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: Annotated[UUID, Path(description="The job's UUID")]):
    """
    Get the current status of a job.
    Args:
        job_id: The job's UUID.
    Returns:
        The job's current status and data if available.
    """
    return jobs.get(job_id, {"status": "not_found"})

async def process(job_id: UUID, path: str, match: int):
    """
    Process a tracking job.
    Args:
        job_id: The job's UUID.
        path: The video file's path.
        match: The match number.
    """
    try:
        def progress_callback(info: dict):
            jobs[job_id]["progress"] = info

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_pipeline, path, match, progress_callback)
        jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        jobs[job_id] = {"status": "error", "result": str(e)}


def detect_digits(crops: list, digit_model: YOLO, conf_thresh=0.4) -> list[str | None]:
    """
    Use batched inference to detect digits on cropped robot images.
    Args:
        crops: A list of images, each cropped to one robot.
        digit_model: A model that detects individual digits.
        conf_thresh: The confidence threshold for allowing digit detections.
    Returns:
        A list containing a string of the digits detected from left to right, for each image.
    """
    if not crops:
        return []

    batch_results = digit_model(crops, conf=conf_thresh, verbose=False, stream=True)

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
    """
    Rank the similarity of detected robot digits with each team's number.
    Args:
        digits: The robot's detected bumper digits.
        teams: A list containing all possible team numbers.
    Returns:
        A numpy array with a similarity score for each possible number,
        calculated using normalized levenshtein distance.
    """
    scores = np.zeros(6)
    for i, team in enumerate(teams):
        dist = Levenshtein.distance(digits, team)
        max_len = max(len(digits), len(team))
        score = 1 - (dist / max_len)
        scores[i] = score
    return scores


def update(track: int, teams: list[str], alliance: int, digits: str, track_votes: dict, track_teams: dict, team_tracks: np.ndarray) -> None:
    """
    Update an unrecognized track's votes and match to a team when possible.
    Args:
        track: The track's ID.
        teams: The numbers of the 6 teams in this match, blue alliance followed by red alliance.
        alliance: The robot's detected alliance color.
        digits: The robot's detected bumper digits.
        track_votes: A state dictionary, accumulating votes to every possible team in a numpy array for each track ID.
        track_teams: A state dictionary, containing each track's chosen team index.
        team_tracks: A state numpy array, containing the current track ID of each team.
    """
    votes = (digit_scores(digits, teams) * DIGIT_WEIGHT) + \
        ((ALLIANCE_MASK == alliance) * ALLIANCE_WEIGHT)
    if track not in track_votes:
        track_votes[track] = np.zeros(6)
    track_votes[track] = ALPHA * track_votes[track] + BETA * votes
    match = np.argmax(track_votes[track] * (team_tracks[np.arange(6)] == 0))
    if track_votes[track][match] >= TEAM_THRESH:
        track_teams[track] = match
        team_tracks[match] = track


def smooth_and_interpolate(positions: list, total_duration_ms: float, target_hz: float = 50, sigma=5):
    """
    Average duplicate detections, interpolate in gaps and smooth with a gaussian filter.
    Args:
        positions: Input positions for each frame with a timestamp.
        total_duration_ms: The total video duration.
        target_hz: Output sample rate.
        sigma: Gaussian filter sigma value.
    Returns:
        Smoothed and interpolated positions, evenly spaced in the target sample rate.
    """
    if not positions:
        return []

    # Average duplicate/close frames (in case a track has multiple hits per frame)
    timestamp_buckets = defaultdict(list)
    for t, x, y in positions:
        timestamp_buckets[t].append((x, y))

    known = {t: np.mean(pts, axis=0) for t, pts in timestamp_buckets.items()}
    known_times = np.array(sorted(known.keys()))

    relative_times = known_times - known_times[0]

    # Generate evenly spaced time points
    num_samples = int((total_duration_ms / 1000) * target_hz)
    sample_times = np.linspace(0, total_duration_ms, num_samples)

    known_coords = np.array([known[t] for t in known_times])

    xs = np.interp(sample_times, known_times, known_coords[:, 0])
    ys = np.interp(sample_times, known_times, known_coords[:, 1])

    # Smooth the signal (only works well if the robot is on screen)
    xs = gaussian_filter1d(xs, sigma=sigma)
    ys = gaussian_filter1d(ys, sigma=sigma)

    return list(zip(xs, ys))


def run_pipeline(video_path: str, match_number: int, progress_callback=None) -> dict:
    """
    Run the tracking pipeline.
    Args:
        video_path: The video file path.
        match_number: The match number.
        progress_callback: A callback for updating status progress, receiving the new info after each iteration.
    Returns:
        A dictionary containing results and video data:
            - 'teams': The list of team numbers for this match, blue alliance followed by red alliance.
            - 'duration': Video duration.
            - 'sample_rate': The rate of output position sampling in Hz.
            - 'frame_width': Video frame width.
            - 'frame_height': Video frame height.
            - 'trajectories': A dictionary containing a list of timestamped positions for each robot.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    cap.grab()
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    teams = schedule[match_number]

    track_votes = {}
    track_teams = {}
    team_tracks = np.zeros(6, dtype=int)

    track_positions = {}


    with tqdm(total=frame_count) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            results = model.track(
                source=frame,
                persist=True,
                tracker=os.path.join(BASE_DIR, "frc_botsort.yaml"),
                verbose=False,
                conf=0.1
            )

            result = results[0]


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
                track_positions[tid].append((timestamp, cx, cy))
                
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
                    "timestamp": timestamp,
                    "total": frame_count,
                    "frame": pbar.n,
                    "it_per_s": round(pbar.format_dict.get("rate") or 0, 2),
                    "elapsed": round(pbar.format_dict.get("elapsed") or 0, 1),
                    "eta": round((frame_count - pbar.n) / pbar.format_dict["rate"], 1)
                        if pbar.format_dict.get("rate") else None,
                })

    cap.release()

    total_ms = duration * 1000

    all_frame_positions = []
    for team_idx in range(len(teams)):
        tids = [tid for tid, idx in track_teams.items() if idx == team_idx]
        positions = []
        for tid in tids:
            positions.extend(track_positions.get(tid, []))
        positions.sort(key=lambda p: p[0])
        all_frame_positions.append(smooth_and_interpolate(positions, total_ms))

    return {
        "teams": teams,
        "duration": duration,
        "sample_rate": SAMPLE_RATE,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "trajectories": {
            teams[i]: [[float(x), float(y)] for x, y in positions]
            for i, positions in enumerate(all_frame_positions)
        }
    }
