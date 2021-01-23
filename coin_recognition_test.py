from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2, sys, os


class CoinRecognition:
    def __init__(self, model_path):
        self.model_path = model_path
        # self.in_path = in_path
        self.values = []

    def recognize(self, cropped):
        # Load the trained neural network
        if getattr(sys, 'frozen', False):
            # image = PhotoImage(file=os.path.join(sys._MEIPASS, "files/bg.png"))
            model = load_model(os.path.join(sys._MEIPASS, self.model_path))

        else:
            model = load_model(self.model_path)

        # Dictionary of coins
        coins_dict = {0: 0.01, 1: 0.02, 2: 0.05, 3: 0.1, 4: 0.2,
                      5: 0.5, 6: 1.0, 7: 2.0, 8: 5.0, 9: "tail"}

        # Load images
        for coin in cropped:
            # Process for classification
            coin_classify = cv2.resize(coin, (28, 28))
            coin_classify = coin_classify.astype("float") / 255.0
            coin_classify = img_to_array(coin_classify)
            coin_classify = np.expand_dims(coin_classify, axis=0)

            # Classify the image
            coin_prob = model.predict(coin_classify)[0]
            max_coin = coin_prob.argmax()

            self.values.append(coins_dict[max_coin])

        return self.values, sum(filter(lambda i: isinstance(i, float), self.values))
