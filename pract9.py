import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("cctv.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("CCTV Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()