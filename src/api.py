import time

from ultralytics import YOLO
import cv2

def read_video_in_batches(video_path, batch_size=5):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Cannot open video")

    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch.append(frame)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

    cap.release()

def inference(model_path, video_path, batch_size=5, show=False):
    model = YOLO(model_path)

    for batch in read_video_in_batches(video_path, batch_size):
        results = model(batch, batch=batch_size , verbose= show)

        if show:
            for r in results:
                annotated = r.plot()
                cv2.imshow("YOLO Inference", annotated)
                time.sleep(0.3)

                # press ESC to quit early
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    return
        yield results

    if show:
        cv2.destroyAllWindows()

# [ Transfer utils ]  push_message ( queue_name ) , get_message( queue_name )

