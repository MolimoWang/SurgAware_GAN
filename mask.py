import os
import nrrd
import numpy as np
from PIL import Image

# 你自己的路径
NRRD_DIR = "all_masks_nrrd"              # 存 tumor_mask_xx.nrrd 的文件夹
OUT_MASK_DIR = "Tr_mask/all_masks_iMRI"  # 输出2D掩码PNG的文件夹
os.makedirs(OUT_MASK_DIR, exist_ok=True)

for fname in os.listdir(NRRD_DIR):
    if not fname.endswith(".nrrd"):
        continue

    nrrd_path = os.path.join(NRRD_DIR, fname)
    print("Loading mask:", nrrd_path)
    mask_vol, _ = nrrd.read(nrrd_path)   # shape 比如 (512,512,348)

    mask_vol = mask_vol.astype(np.float32)
    mask_vol = (mask_vol > 0).astype(np.uint8)  # 二值化 0/1

    H, W, D = mask_vol.shape

    # 举例：把所有有肿瘤的层都导出
    for k in range(D):
        if not np.any(mask_vol[:, :, k]):
            continue  # 这一层没肿瘤，跳过

        mask_slice = mask_vol[:, :, k] * 255  # 0/255 uint8
        img = Image.fromarray(mask_slice, mode="L")

        # 这里的命名规则你按自己来改，只要之后能从 iMRI 名字推回 mask 名字就行
        out_name = f"mask_slice_{k:03d}.png"
        out_path = os.path.join(OUT_MASK_DIR, out_name)
        img.save(out_path)
        print("Saved:", out_path)
