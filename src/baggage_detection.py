from ultralytics import YOLO
import cv2
import math

# ==========================
# Load YOLO Model
# ==========================
model = YOLO("yolov8n.pt")

# ==========================
# Load Video
# ==========================
video = cv2.VideoCapture("../videos/baggage.mp4")

if not video.isOpened():
    print("Video not found!")
    exit()

print("Video opened successfully")

# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[0, 24, 26, 28]
    )

    annotated_frame = frame.copy()

    person_count = 0
    bag_count = 0

    person_centers = []
    bag_centers = []

    # ==========================
    # Detection Loop
    # ==========================
    for box in results[0].boxes:

        class_id = int(box.cls[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # ==========================
        # PERSON
        # ==========================
        if class_id == 0:

            person_count += 1

            person_centers.append(
                (center_x, center_y)
            )

            color = (0, 255, 0)
            label = "PERSON"

        # ==========================
        # BAG
        # ==========================
        else:

            bag_count += 1

            bag_centers.append(
                (center_x, center_y)
            )

            color = (0, 0, 255)
            label = "BAG"

        # Draw Bounding Box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw Label
        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ==========================
    # Unattended Bag Detection
    # ==========================
    unattended_bag = False

    for bag_x, bag_y in bag_centers:

        nearest_distance = 99999

        for person_x, person_y in person_centers:

            distance = math.sqrt(
                (bag_x - person_x) ** 2 +
                (bag_y - person_y) ** 2
            )

            if distance < nearest_distance:
                nearest_distance = distance

        if nearest_distance > 250:

            unattended_bag = True

            cv2.circle(
                annotated_frame,
                (bag_x, bag_y),
                25,
                (0, 0, 255),
                3
            )

    # ==========================
    # Dashboard
    # ==========================
    cv2.putText(
        annotated_frame,
        f"Persons: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Bags: {bag_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if bag_count > 0:

        cv2.putText(
            annotated_frame,
            "BAG MONITORING ACTIVE",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    # ==========================
    # ALERT
    # ==========================
    if unattended_bag:

        cv2.putText(
            annotated_frame,
            "UNATTENDED BAG ALERT",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ==========================
    # Display
    # ==========================
    cv2.imshow(
        "RailGuard AI - Baggage Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================
video.release()
cv2.destroyAllWindows()