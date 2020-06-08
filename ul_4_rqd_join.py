import pandas as pd
import cv2
from file_utils import find_files_glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle


def extract_depth(filepath):
    p = Path(filepath).stem
    return float(p[10:])


hole_ids = [1, 3]

for hole in hole_ids:

    image_paths = find_files_glob('out/test_images_pad/', extension=f'KAD17_{hole:03d}*.jpg')

    p = Path(image_paths[0]).stem

    files = pd.DataFrame()
    files['files'] = image_paths
    files['depth'] = files['files'].apply(extract_depth)
    files.sort_values(by='depth', inplace=True)

    detectors = np.arange(0, 145)
    pixels = [[] for detector in detectors]

    for idx, row in files.iterrows():
        img = cv2.imread(row['files'], cv2.IMREAD_GRAYSCALE)

        if img.shape != (145, 145):
            continue

        for idx, detector in enumerate(detectors):
            pixels[idx].extend(img[detector, :])

        # cv2.imshow('', img)
        # cv2.waitKey(0)

    pixels = np.array(pixels)
    with open(f'out/KAD17-{hole:03d}.pkl') as file:
        pickle.dump(pixels, file)

    df = pd.read_pickle('out/merged_master.pkl')
    df = df[df['Hole'] == f'KAD17-{hole:03d}']

    plt.figure(figsize=(96, 16))
    plt.subplot(2, 1, 1)
    plt.imshow(pixels, aspect='auto')
    plt.ylabel('Vertical Pixels')
    plt.xlabel('Depth (pixels)')

    plt.subplot(2, 1, 2)
    plt.fill_between(df['From_m'], df['RQD_m'])
    plt.xlabel('Depth (m)')
    plt.ylabel('RQD')
    plt.xlim([min(files['depth']), max(files['depth'])])

    plt.savefig(f'out/KAD17-{hole:03d}.png')
    plt.show()

