# This file is based on the example from https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html

from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.io import read_video

plt.rcParams["savefig.bbox"] = "tight"

class RaftModel:
    def __preprocess(self, batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
                T.Resize(size=(360, 640)),
            ]
        )
        batch = transforms(batch)
        return batch

    def __plot(self, imgs, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img = F.to_pil_image(img.to("cpu"))
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()
        plt.show()

    def __init__(self, device):
        self.device = device
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)

    def predict(self, base_images, subsequent_images):
        base_image_batch = self.__preprocess(base_images)
        subsequent_image_batch = self.__preprocess(subsequent_images)

        flow_batch = self.model(base_image_batch.to(self.device), subsequent_image_batch.to(self.device))

        return flow_batch
    
    def visualize(self, base_image_batch, flow_batch):
        flow_imgs = flow_to_image(flow_batch)

        grid = [[base_image, flow_img] for (base_image, flow_img) in zip(base_image_batch, flow_imgs)]
        self.__plot(grid)

        return
    
    def test(self):
        video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
        video_path = Path(tempfile.mkdtemp()) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)
        frames, _, _ = read_video(str(video_path))

        img1_batch = torch.stack([frames[100], frames[150]])
        img2_batch = torch.stack([frames[101], frames[151]])

        list_of_flows = self.predict(img1_batch, img2_batch)
        flow_batch = list_of_flows[-1]

        self.visualize(img1_batch, flow_batch)

        return



