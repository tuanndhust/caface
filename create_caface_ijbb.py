import os
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.utils.face_align import norm_crop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ijb_root",
    type=str,
    required=True,
    help="Path đến thư mục IJBB (chứa loose_crop/ và meta/)"
)
parser.add_argument(
    "--output_root",
    type=str,
    required=True,
    help="Nơi tạo thư mục IJB đúng chuẩn CAFace"
)
args = parser.parse_args()

ijb_root = args.ijb_root
loose_dir = os.path.join(ijb_root, "loose_crop")
meta_dir_src = os.path.join(ijb_root, "meta")

# OUTPUT STRUCTURE
ijb_out = os.path.join(args.output_root, "IJB")
aligned_out = os.path.join(ijb_out, "aligned")
helper_root = os.path.join(ijb_out, "insightface_helper")
ijb_protocol_out = os.path.join(helper_root, "ijb")
ijb_meta_out = os.path.join(helper_root, "meta")

os.makedirs(aligned_out, exist_ok=True)
os.makedirs(ijb_protocol_out, exist_ok=True)
os.makedirs(ijb_meta_out, exist_ok=True)

# ----------------------------
# 1. ALIGN IMAGES
# ----------------------------
print(">>> Loading detector + alignment model...")
# app = insightface.app.FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
# app.prepare(ctx_id=0)
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640,640))


img_list = sorted([f for f in os.listdir(loose_dir) if f.endswith(".jpg")])

img2path_lines = []

print(">>> Aligning {} images...".format(len(img_list)))

for name in tqdm(img_list):
    img_id = os.path.splitext(name)[0]
    path = os.path.join(loose_dir, name)

    img = cv2.imread(path)
    if img is None:
        continue

    faces = app.get(img)
    if len(faces) == 0:
        continue

    kps = faces[0].kps
    aligned = norm_crop(img, kps, image_size=112)

    out_path = os.path.join(aligned_out, f"{img_id}.jpg")
    cv2.imwrite(out_path, aligned)

    img2path_lines.append(f"{img_id} {out_path}\n")

# ----------------------------
# 2. COPY PROTOCOL FILES
# ----------------------------
protocol_files = [
    "ijbb_face_tid_mid.txt",
    "ijbb_template_pair_label.txt",
]

print(">>> Copying IJB-B protocol files...")

for f in protocol_files:
    src = os.path.join(meta_dir_src, f)
    dst = os.path.join(ijb_protocol_out, f)

    if os.path.exists(src):
        os.system(f"cp {src} {dst}")
    else:
        print("WARNING: không tìm thấy", src)

# ----------------------------
# 3. GENERATE CAFace META FILES
# ----------------------------
print(">>> Generating CAFace meta files...")

# img2path.txt
with open(os.path.join(ijb_meta_out, "img2path.txt"), "w") as f:
    f.writelines(img2path_lines)

# face2img + template2img_list
face_tid_mid_path = os.path.join(meta_dir_src, "ijbb_face_tid_mid.txt")

# face2img
with open(face_tid_mid_path, "r") as f:
    lines = f.readlines()

with open(os.path.join(ijb_meta_out, "face2img.txt"), "w") as f:
    for line in lines:
        face_id, template_id, img_id = line.strip().split()
        f.write(f"{face_id} {img_id}\n")

# template2img_list
template_map = {}

for line in lines:
    face_id, template_id, img_id = line.strip().split()
    template_id = int(template_id)
    template_map.setdefault(template_id, []).append(img_id)

with open(os.path.join(ijb_meta_out, "template2img_list.txt"), "w") as f:
    for tid, imgs in template_map.items():
        f.write(f"{tid} {' '.join(imgs)}\n")

print("\n>>> DONE! CAFace IJB structure created successfully.")
print("Output folder:", ijb_out)
