"""HuggingFace dataset wrapper for DDM4IP."""
import warnings
from typing import Tuple

import torch
import torch.utils.data
import torchvision.transforms.v2 as v2

from ddm4ip.data.utils import AddLocMapTransform
from ddm4ip.utils import distributed
from .base import Batch, DatasetType, Datasplit


class HuggingFaceDataset(torch.utils.data.Dataset[Batch], DatasetType):
    """Dataset wrapper for HuggingFace datasets.
    
    Supports datasets with image triplets (drop, blur, clear) like RaindropClarity.
    """
    def __init__(
        self,
        path,
        degradation,
        split: Datasplit,
        dset_cfg,
        shuffle_clean: bool = True,
        generator=None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace datasets. "
                "Install it with: pip install datasets"
            )
        
        self.split = split
        self.shuffle_clean = shuffle_clean
        self.corruption = degradation
        self.conditional = dset_cfg.get("cond", False)
        self.max_imgs = dset_cfg.get("max_imgs", None)
        self.x_flip = dset_cfg.get("x_flip", False) and self.split == Datasplit.TRAIN
        self.space_conditioning = dset_cfg.get("space_conditioning", False)
        
        # HuggingFace specific parameters
        dataset_name = dset_cfg.get("hf_dataset_name", path)
        config_name = dset_cfg.get("hf_config", None)
        hf_split = dset_cfg.get("hf_split", split.value)
        
        # Image field names in the HuggingFace dataset
        self.clean_field = dset_cfg.get("clean_field", "clear")  # background focused
        self.corrupt_field = dset_cfg.get("corrupt_field", "drop")  # raindrop focused
        
        # Load dataset from HuggingFace
        distributed.print0(f"Loading HuggingFace dataset '{dataset_name}' (config='{config_name}', split='{hf_split}')...")
        self.hf_dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=hf_split,
        )
        
        # Apply max_imgs limit
        if self.max_imgs is not None and self.max_imgs < len(self.hf_dataset):
            self.hf_dataset = self.hf_dataset.select(range(self.max_imgs))
        
        # Setup transforms
        img_transform = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if self.space_conditioning:
            img_transform.append(AddLocMapTransform())
        if self.x_flip:
            img_transform.append(v2.RandomHorizontalFlip(p=0.5))
        
        self.img_transform = v2.Compose(img_transform)
        self.label_dim = 0  # No labels for this dataset
        
        # Setup clean/corrupt index mapping for shuffling
        if shuffle_clean:
            self.clean_ids = torch.randperm(len(self.hf_dataset), generator=generator)
        else:
            self.clean_ids = torch.arange(len(self.hf_dataset))
        
        # Determine image channels and size from a sample
        sample_batch = self[0]
        sample_clean_cond = sample_batch.clean_conditioning
        sample_clean_img = sample_batch.clean
        assert sample_clean_img is not None
        self.clean_img_size = (sample_clean_img.shape[-3], sample_clean_img.shape[-2], sample_clean_img.shape[-1])
        self.clean_conditioning_channels = sample_clean_cond.shape[0] if sample_clean_cond is not None else 0
        
        sample_corrupt_cond = sample_batch.corrupt_conditioning
        sample_corrupt_img = sample_batch.corrupt
        assert sample_corrupt_img is not None
        self.corrupt_img_size = (sample_corrupt_img.shape[-3], sample_corrupt_img.shape[-2], sample_corrupt_img.shape[-1])
        self.corrupt_conditioning_channels = sample_corrupt_cond.shape[0] if sample_corrupt_cond is not None else 0
        
        distributed.print0(f"Loaded HuggingFace dataset '{dataset_name}':")
        distributed.print0(f"Config:                        {config_name}")
        distributed.print0(f"Split:                         {hf_split}")
        distributed.print0(f"Number of images:              {len(self)}")
        distributed.print0(f"Label dimension:               {self.label_dim}")
        distributed.print0(f"Clean image size:              {self.clean_img_size}")
        distributed.print0(f"Corrupt image size:            {self.corrupt_img_size}")
        distributed.print0(f"Space conditioning:            {self.space_conditioning}")
        distributed.print0(f"Clean conditioning channels:   {self.clean_conditioning_channels}")
        distributed.print0(f"Corrupt conditioning channels: {self.corrupt_conditioning_channels}")
        distributed.print0()
    
    def get_conditioning(self, img: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Extract conditioning information from image tensor."""
        cond = None
        if self.space_conditioning and img is not None:
            cond = img[..., -2:, :, :]
            img = img[..., :-2, :, :]
        return cond, img
    
    def add_corruption(self, img, conditioning):
        """Apply corruption/degradation to image."""
        if self.corruption is None:
            return img
        corrupt_img = self.corruption(
            img.unsqueeze(0), conditioning=conditioning,
        ).squeeze(0).clamp(0, 1)
        return corrupt_img
    
    def __getitem__(self, index):
        # Get clean and corrupt indices
        clean_idx = int(self.clean_ids[index % len(self)])
        noisy_idx = index % len(self)
        
        # Load images from HuggingFace dataset
        clean_sample = self.hf_dataset[clean_idx]
        noisy_sample = self.hf_dataset[noisy_idx]
        
        # Extract and transform clean image
        clean_img = clean_sample[self.clean_field]
        clean_img = self.img_transform(clean_img)
        clean_conditioning, clean_img = self.get_conditioning(clean_img)
        
        # Extract and transform corrupt image
        noisy_img = noisy_sample[self.corrupt_field]
        noisy_img = self.img_transform(noisy_img)
        noisy_conditioning, noisy_img = self.get_conditioning(noisy_img)
        noisy_img = self.add_corruption(noisy_img, noisy_conditioning)
        
        return Batch(
            clean=clean_img,
            corrupt=noisy_img,
            clean_label=None,
            corrupt_label=None,
            noise_level=self.noise_level,
            clean_conditioning=clean_conditioning,
            corrupt_conditioning=noisy_conditioning,
        )
    
    @property
    def noise_level(self) -> torch.Tensor:
        """Return noise level from corruption model."""
        if self.corruption is None:
            return torch.tensor(0.0)
        try:
            return self.corruption.noise_model.sigma
        except AttributeError:
            return torch.tensor(0.0)
    
    def __len__(self):
        return len(self.hf_dataset)
