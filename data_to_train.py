import subprocess
import sys
import os


class PopulateTrainData():
    def __init__(self, coin_path):
        self.coin = coin_path

    def populate(self):
        in_path = "img/test/" + self.coin
        out_path = "img/train/" + self.coin
        only_files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]

        for i in range(len(only_files)):
            print(only_files[i])
            subprocess.call([sys.executable or 'python', 'coin_extraction.py', '-i', in_path + only_files[i], '-o', out_path])
