# AutoScout

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-02569B?logo=flutter&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-111F68?logo=ultralytics&logoColor=white)

> A tool for automatically tracking FRC robots from match footage

AutoScout is a computer vision pipeline for automatically tracking and identifying all six robots in FRC match footage. It combines YOLO-based robot detection with ByteTrack multi-object tracking and bumper number recognition to assign each robot to its team throughout a match, outputting smoothed trajectories for post-match analysis.

## Credits
### Datasets

**Robot Detection**: [Robot Detection](https://universe.roboflow.com/main-wcgiu/robot-detection-xru6m) by Roboflow user `main-wcgiu`, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).