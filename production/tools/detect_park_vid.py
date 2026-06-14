import os
import cv2
import numpy as np

def get_green_ratio(path):
    cap = cv2.VideoCapture(path)
    # read frame at 10% progress to avoid dark/black starts
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.1))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return 0.0
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    # Green range in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return float(np.sum(mask > 0) / mask.size)

d = "production/_media"
vids = [f for f in os.listdir(d) if f.endswith(".mp4") and not f.startswith("out_")]
for v in sorted(vids):
    path = os.path.join(d, v)
    ratio = get_green_ratio(path)
    print(f"{v}: green_ratio = {ratio:.3f}")
