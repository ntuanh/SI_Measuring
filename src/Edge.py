import numpy as np
import cv2 , time

from src.api import inference , read_video_in_batches

class Edge:
    def __init__(self , config ):
        self.model = config["model"]["name"]
        self.batch_size = config["model"]["batch"]
        self.data = config["model"]["data"]
        self.show = config["visual"]["enable"]

        if config["mode"]["only"] == "Edge":
            self.enable_inference = True  # inference on edge else on cloud
        else:
            self.enable_inference = False

    def run(self):
        print("EDGE are running")
        print(f"mode {self.enable_inference}")
        # Cloud mode
        if self.enable_inference == False:
            for batch in read_video_in_batches(self.data , self.batch_size):
                yield batch
        else :
            for results in inference(self.model, self.data, self.batch_size, show=self.show):
                yield  results
