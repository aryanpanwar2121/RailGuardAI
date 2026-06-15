from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("../videos/crowd.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "RailGuard AI - Person Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()