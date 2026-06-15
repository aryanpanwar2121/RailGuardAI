from ultralytics import YOLO
import cv2
from datetime import datetime
import os

# ==========================
# Load Model
# ==========================
model = YOLO("yolov8n.pt")

# ==========================
# Load Video
# ==========================
video = cv2.VideoCapture("../videos/crowd.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

# ==========================
# Setup
# ==========================
alerted_ids = set()

os.makedirs("../outputs", exist_ok=True)

log_file = "../outputs/intrusion_log.txt"

# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = video.read()

    if not ret:
        break

    # Person Tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = frame.copy()

    height, width = frame.shape[:2]

    # Track Zone (Bottom 40%)
    track_y = int(height * 0.60)

    # ==========================
    # Red Transparent Track Zone
    # ==========================
    overlay = annotated_frame.copy()

    cv2.rectangle(
        overlay,
        (0, track_y),
        (width, height),
        (0, 0, 255),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.25,
        annotated_frame,
        0.75,
        0,
        annotated_frame
    )

    cv2.line(
        annotated_frame,
        (0, track_y),
        (width, track_y),
        (0, 0, 255),
        3
    )

    # ==========================
    # Tracking Logic
    # ==========================
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            center_y = (y1 + y2) // 2

            # Draw Bounding Box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Show Person ID
            cv2.putText(
                annotated_frame,
                f"ID:{person_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ==========================
            # Track Intrusion Detection
            # ==========================
            if center_y > track_y:

                cv2.putText(
                    annotated_frame,
                    "TRACK INTRUSION ALERT",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # Avoid duplicate alerts
                if person_id not in alerted_ids:

                    alerted_ids.add(person_id)

                    timestamp = datetime.now()

                    filename = (
                        "../outputs/intrusion_"
                        + timestamp.strftime("%Y%m%d_%H%M%S")
                        + ".png"
                    )

                    # Save Evidence Screenshot
                    cv2.imwrite(
                        filename,
                        annotated_frame
                    )

                    # Log Intrusion
                    with open(log_file, "a") as file:

                        file.write(
                            f"{timestamp} | Person ID {person_id} | Track Intrusion\n"
                        )

                    print(
                        f"ALERT: Person {person_id} entered track zone"
                    )

    # ==========================
    # Display
    # ==========================
    cv2.imshow(
        "RailGuard AI - Person Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================
video.release()
cv2.destroyAllWindows()