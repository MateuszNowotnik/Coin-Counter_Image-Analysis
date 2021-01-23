import subprocess
import sys
import os


def main():
    coin = "tail/"

    in_path = "img/test/" + coin
    out_path = "img/train/" + coin
    only_files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]

    for i in range(len(only_files)):
        print(only_files[i])
        subprocess.call([sys.executable or 'python', 'coin_extraction.py', '-i', in_path + only_files[i], '-o', out_path])


if __name__ == '__main__':
    main()
