import cv2, sys, os
import numpy as np
import torch


class CoinRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        # self.in_path = in_path
        self.coin_classify = []
        self.values = []

    def recognize(self, cropped):
        # Importing a custom model
        if getattr(sys, 'frozen', False):
            # model = load_model(os.path.join(sys._MEIPASS, self.model_path))
            # model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=self.model_path)
            model = torch.hub.load(os.path.join(sys._MEIPASS, 'ultralytics/yolov5', 'custom', path_or_model=self.model_path))
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=self.model_path)
            # model = load_model(self.model_path)

        # Dictionary of coins
        coins_dict = {0: 0.10, 1: 0.01, 2: 1.00, 3: 0.20, 4: 0.02,
                      5: 2.00, 6: 0.50, 7: 0.05, 8: 5.0, 9: "Tail"}

        # Load images
        for coin in cropped:
            coin = cv2.resize(coin, (416, 416))
            # Classify the image
            results = model(coin, size=416)
            # If detected
            # print(results.xyxy[0])
            if results.xyxy[0].numpy().size != 0:
                coin_classify = results.xyxy[0].data[0][5].numpy()
                print(coin_classify)
                # Add to the sum
                self.values.append(coins_dict[int(coin_classify)])
            else:
                self.values.append("Unknown")

        # Return values and sum of floats
        return self.values, sum(filter(lambda i: isinstance(i, float), self.values))
