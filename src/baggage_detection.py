from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("../videos/baggage.mp4")
if not video.isOpened():
    print("Video not found!")
    exit()

print("Video opened successfully")

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = frame.copy()

    for box in results[0].boxes:

        cls = int(box.cls[0])

        name = model.names[cls]

        if name in ["backpack", "handbag", "suitcase"]:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"BAG: {name}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    cv2.imshow(
        "RailGuard AI - Baggage Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()