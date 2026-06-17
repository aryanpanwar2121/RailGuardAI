from ultralytics import YOLO
import cv2
import math
import time

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
# Smart Bag Tracking
# ==========================

bag_owner_map = {}

bag_last_seen_with_owner = {}

alert_timeout = 5

# ==========================
# Main Loop
# ==========================

while True:

    ret, frame = video.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0, 24, 26, 28]
    )

    annotated_frame = frame.copy()

    person_count = 0
    bag_count = 0

    person_centers = []

    bag_objects = []

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

        track_id = (
            int(box.id[0])
            if box.id is not None
            else 0
        )

        # ==========================
        # PERSON
        # ==========================

        if class_id == 0:

            person_count += 1

            person_centers.append(
                (
                    center_x,
                    center_y,
                    track_id
                )
            )

            color = (0, 255, 0)

            label = f"PERSON {track_id}"

        # ==========================
        # BAG
        # ==========================

        else:

            bag_count += 1

            bag_objects.append(
                (
                    center_x,
                    center_y,
                    bag_count
                )
            )

            color = (0, 0, 255)

            label = "BAG"

        # ==========================
        # Draw Box
        # ==========================

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

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
    # Smart Owner Mapping
    # ==========================

    unattended_bag = False

    for bag_x, bag_y, bag_id in bag_objects:

        nearest_distance = 99999

        owner_id = None

        for person_x, person_y, person_id in person_centers:

            distance = math.sqrt(
                (bag_x - person_x) ** 2 +
                (bag_y - person_y) ** 2
            )

            if distance < nearest_distance:

                nearest_distance = distance

                owner_id = person_id

        # First Association

        if bag_id not in bag_owner_map:

            bag_owner_map[bag_id] = owner_id

        # Owner Near Bag

        if nearest_distance < 250:

            bag_last_seen_with_owner[
                bag_id
            ] = time.time()

        # Owner Away

        else:

            if bag_id not in bag_last_seen_with_owner:

                bag_last_seen_with_owner[
                    bag_id
                ] = time.time()

            elapsed_time = (
                time.time()
                -
                bag_last_seen_with_owner[
                    bag_id
                ]
            )

            if elapsed_time > alert_timeout:

                unattended_bag = True

                cv2.circle(
                    annotated_frame,
                    (bag_x, bag_y),
                    30,
                    (0, 0, 255),
                    4
                )

        # Owner Label

        cv2.putText(
            annotated_frame,
            f"Owner:{bag_owner_map[bag_id]}",
            (bag_x, bag_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
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

    cv2.putText(
        annotated_frame,
        "OWNER MAPPING ACTIVE",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        "BAG MONITORING ACTIVE",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # ==========================
    # Alert
    # ==========================

    if unattended_bag:

        cv2.putText(
            annotated_frame,
            "UNATTENDED BAG ALERT",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ==========================
    # Display
    # ==========================

    cv2.imshow(
        "RailGuard AI - Smart Baggage Monitoring",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================

video.release()

cv2.destroyAllWindows()