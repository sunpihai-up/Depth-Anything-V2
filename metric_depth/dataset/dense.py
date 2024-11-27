import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop


class Dense(Dataset):
    def __init__(self, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size

        with open(filelist_path, "r") as f:
            self.filelist = f.read().splitlines()

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True if mode == "train" else False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

    def prepare_depth(self, depth, reg_factor, d_max):
        # Normalize depth
        depth = np.clip(depth, 0.0, d_max)
        depth = depth / d_max
        depth = np.log(depth + 1e-6) / reg_factor + 1.0
        depth = depth.clip(0.0, 1.0)
        return depth

    def __getitem__(self, item):
        img_path = self.filelist[item].split(" ")[0]
        depth_path = self.filelist[item].split(" ")[1]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Normalized log depth
        reg_factor, d_max = 6.2044, 1000
        depth = np.load(depth_path)
        # depth = self.prepare_depth(depth, reg_factor, d_max)

        sample = self.transform({"image": image, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        sample["valid_mask"] = sample["depth"] >= 0
        sample["image_path"] = img_path

        return sample

    def __len__(self):
        return len(self.filelist)
