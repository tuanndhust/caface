import os
import cv2
import numpy as np
from skimage import transform as trans

SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)


def align_face(img, landmarks):
    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, SRC)
    M = tform.params[0:2]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return aligned


meta_file = "meta/ijbb_name_5pts_score.txt"
img_root = "loose_crop"
save_root = "IJBB_aligned"

os.makedirs(save_root, exist_ok=True)

with open(meta_file) as f:
    for line in f:
        parts = line.strip().split()
        img_name = parts[0]

        pts = np.array(list(map(float, parts[1:11]))).reshape(5, 2)

        img_path = os.path.join(img_root, img_name)
        out_path = os.path.join(save_root, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        aligned = align_face(img, pts)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, aligned)
