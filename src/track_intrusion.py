from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# =========================
# Setup
# =========================

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("../videos/crowd.mp4")

if not video.isOpened():
    print("ERROR: Video not found!")
    exit()

os.makedirs("../outputs", exist_ok=True)

saved_recently = False

# =========================
# Main Loop
# =========================

while True:

    ret, frame = video.read()

    if not ret:
        print("Video Ended")
        break

    results = model(frame, classes=[0])

    annotated_frame = frame.copy()

    height, width, _ = frame.shape

    # =========================
    # Track Zone
    # =========================

    track_y = int(height * 0.90)

    overlay = annotated_frame.copy()

    cv2.rectangle(
        overlay,
        (0, track_y),
        (width, height),
        (0, 0, 255),
        -1
    )

    alpha = 0.25

    cv2.addWeighted(
        overlay,
        alpha,
        annotated_frame,
        1 - alpha,
        0,
        annotated_frame
    )

    cv2.rectangle(
        annotated_frame,
        (0, track_y),
        (width, height),
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "TRACK AREA",
        (20, track_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # =========================
    # Crowd Count
    # =========================

    person_count = len(results[0].boxes)

    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    if person_count > 5:
        cv2.putText(
            annotated_frame,
            "HIGH CROWD ALERT",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =========================
    # Intrusion Detection
    # =========================

    intrusion = False

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            "Person",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        person_bottom = y2

        if person_bottom > track_y:
            intrusion = True

    # =========================
    # Alert + Screenshot
    # =========================

    if intrusion:

        cv2.putText(
            annotated_frame,
            "TRACK INTRUSION ALERT",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if not saved_recently:

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"../outputs/intrusion_{timestamp}.png"

            cv2.imwrite(filename, annotated_frame)

            print(f"Screenshot Saved: {filename}")

            saved_recently = True

    else:
        saved_recently = False

    # =========================
    # Timestamp
    # =========================

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cv2.putText(
        annotated_frame,
        current_time,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # =========================
    # Display
    # =========================

    cv2.imshow("RailGuard AI - Track Safety", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()