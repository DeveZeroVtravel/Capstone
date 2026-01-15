import cv2
import numpy as np
from collections import deque
from MyYOLO import MyYOLO
from Alert import AlertSystem

alert = AlertSystem('./beep-warning-6387.mp3')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 30)
myyolo = MyYOLO('./DriverDrowsiness/yolo11n_training2/weights/best_ncnn_model')

FPS = 30
WINDOW_10S = int(10 * FPS)
WINDOW_30S = int(30 * FPS)
FATIGUE_THRESHOLD = 0.7

fatigue_history = deque(maxlen=WINDOW_30S)

warning_active = False
sound_active = False

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    result, confidence = myyolo.predict(frame)
    
    no_human = (result == "non-humans")
    
    
    
    is_fatigue = 1 if result == "fatigue" or result == "non-humans" else 0
    fatigue_history.append(is_fatigue)
    frames_10s = list(fatigue_history)[-WINDOW_10S:]
    frames_30s = list(fatigue_history)
    fatigue_ratio_10s = sum(frames_10s) / len(frames_10s) if len(frames_10s) > 0 else 0
    fatigue_ratio_30s = sum(frames_30s) / len(frames_30s) if len(frames_30s) > 0 else 0
    warning_active = len(frames_10s) >= WINDOW_10S and fatigue_ratio_10s >= FATIGUE_THRESHOLD
    sound_active = len(frames_30s) >= WINDOW_30S and fatigue_ratio_30s >= FATIGUE_THRESHOLD
    
    if no_human and confidence > 0.9:
        warning_active = True
        sound_active = True
    cv2.putText(frame, str(result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, str(round(confidence, 2)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    frame = alert.dispWarn(frame, 1 if warning_active else 0)
    alert.playsound(1 if sound_active else 0)
    
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        alert.playsound(0)
        fatigue_history.clear()
        warning_active = False
        sound_active = False

alert.stop()
cam.release()
cv2.destroyAllWindows()