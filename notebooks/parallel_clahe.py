import os
import sys
import time
sys.path.append('../src')

from clahe import run_clahe
from PIL import Image
from multiprocessing import Pool

data_dir = '../data/COVID-19 Radiography Database'
output_dir = '../data/clahe_applied/'


def apply_clahe(img_path):
    dirs = [output_dir] + os.path.normpath(img_path).split(os.sep)[-2:]
    output_path = os.path.join(*dirs)
    if os.path.isfile(output_path):
        return
    equalized = run_clahe(img_path)
    im = Image.fromarray(equalized)
    im.save(output_path)


if __name__ == '__main__':
    print(os.cpu_count())
    all_paths = [os.path.join(data_dir, 'COVID-19', x) for x in os.listdir(os.path.join(data_dir, 'COVID-19'))] + [
        os.path.join(data_dir, 'NORMAL', x) for x in os.listdir(os.path.join(data_dir, 'NORMAL'))]
    st = time.time()
    with Pool(4) as pool:
        pool.map(apply_clahe, all_paths[:4])
    print(time.time() - st)
