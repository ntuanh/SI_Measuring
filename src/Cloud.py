from ultralytics import YOLO
import cv2 , time

class Cloud:
    def __init__(self , config ):
        self.model = config["model"]["name"]
        self.batch_size = config["model"]["batch"]
        self.data = config["model"]["data"]
        self.fps = config["visual"]["FPS"]
        self.show = config["visual"]["enable"]

        if config["mode"]["only"] == "Cloud":
            self.enable_inference = True  # inference on edge else on cloud
        else :
            self.enable_inference = False

        self.lst_fps = []
        self.prev_time = -1

    def run(self , data ):
        if self.enable_inference :
            model = YOLO(self.model)
            results = model(data, batch=self.batch_size , verbose = self.show)

            if self.show:
                for r in results:
                    annotated = r.plot()
                    cv2.imshow("CLOUD", annotated)
                    time.sleep(0.1)

                    # press ESC to quit early
                    if cv2.waitKey(1) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        return
        else :
            if self.show:
                for r in data:
                    annotated = r.plot()
                    cv2.imshow("CLOUD", annotated)
                    time.sleep(1 / self.fps)

                    # press ESC to quit early
                    if cv2.waitKey(1) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        return

        if self.prev_time == -1 :
            self.prev_time = time.time_ns()
        else :
            period = time.time_ns() - self.prev_time
            fps_mean = period / self.batch_size
            self.lst_fps.append(fps_mean)
            print(f"fps mean of batch {1 / (fps_mean / 1e9)}")




