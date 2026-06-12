from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("../models/yolov8n.pt")

# Open video
video = cv2.VideoCapture("../videos/crowd.mp4")

if not video.isOpened():
    print("ERROR: Video not found!")
    exit()

while True:

    ret, frame = video.read()

    if not ret:
        print("Video ended")
        break

    # Detect only persons
    results = model(frame, classes=[0])

    # Copy frame
    annotated_frame = frame.copy()

    # Draw boxes
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

    # Count persons
    person_count = len(results[0].boxes)

    # Display count
    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Crowd alert
    if person_count > 5:
        cv2.putText(
            annotated_frame,
            "HIGH CROWD ALERT",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("RailGuard AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()