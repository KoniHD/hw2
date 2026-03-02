import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from PIL import Image
import numpy as np


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file: str, root_dir: str, transform=None) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0).astype(np.float32)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.key_pts_frame)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = np.array(Image.open(image_name).convert("L")).astype(np.float32)

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FacialKeypointsHeatmapDataset(Dataset):
    """Face Landmarks dataset with heatmap generation."""

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        output_size: int = 64,
        sigma: float = 1,
    ) -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            output_size (int): Size of the output heatmaps (default: 64x64)
            sigma (float): Standard deviation for Gaussian kernel (default: 1)
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0).astype(np.float32)
        self.root_dir = root_dir
        self.transform = transform
        self.output_size = output_size
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.key_pts_frame)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = np.array(Image.open(image_name).convert("L")).astype(np.float32)

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        # Generate heatmaps
        heatmaps = self.generate_heatmaps(sample["keypoints"])
        sample["heatmaps"] = heatmaps

        return sample

    def generate_heatmaps(self, keypoints) -> torch.Tensor:
        """
        Generate heatmaps for each keypoint
        Args:
            keypoints: Tensor or numpy array of shape (68, 2) for 68 keypoints with (x, y) coordinates
        Returns:
            heatmaps: Tensor of shape (68, output_size, output_size)
        """
        # Convert keypoints to numpy if it's a tensor
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.numpy()

        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros(
            (num_keypoints, self.output_size, self.output_size), dtype=np.float32
        )

        # keypoints are in [-1, 1], map them to [0, output_size - 1]
        keypoints_scaled = (keypoints + 1.0) / 2.0 * (self.output_size - 1)

        # Generate a heatmap for each keypoint
        for i in range(num_keypoints):
            # Get the scaled coordinates
            x, y = keypoints_scaled[i]

            # Skip if keypoint is invalid
            if np.isnan(x) or np.isnan(y):
                continue

            # Convert to int for indexing
            x_int, y_int = (
                max(0, min(self.output_size - 1, int(x))),
                max(0, min(self.output_size - 1, int(y))),
            )

            # Create a single hot pixel
            heatmap = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            heatmap[y_int, x_int] = 1.0

            # Apply gaussian filter to create a soft heatmap
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)

            # Normalize heatmap to [0, 1] range
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            heatmaps[i] = heatmap

        return torch.from_numpy(heatmaps)
