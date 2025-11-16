# HuggingFace Dataset Support

This document describes how to use HuggingFace datasets with DDM4IP, specifically the RaindropClarity dataset.

## Overview

The RaindropClarity dataset is available on HuggingFace at [JacobLinCool/RaindropClarity](https://huggingface.co/datasets/JacobLinCool/RaindropClarity) and contains image triplets for raindrop removal tasks:

- **drop**: Raindrop-focused image (used as corrupt/degraded image)
- **blur**: Raindrop-focused image without raindrop
- **clear**: Background focused image (used as clean/target image)

The dataset has two configurations:
- **day**: Images captured during daytime
- **night**: Images captured during nighttime

## Usage

### Using Pre-configured Datasets

The easiest way to use the RaindropClarity dataset is to use the pre-configured dataset files:

```bash
# For day configuration
python main.py dataset=raindrop_clarity_day ...

# For night configuration
python main.py dataset=raindrop_clarity_night ...
```

### Custom Configuration

You can also create custom configurations by creating a YAML file in `ddm4ip/configs/dataset/`:

```yaml
# custom_raindrop.yaml
defaults:
  - degradation: no_degradation
  - noise: no_noise

name: huggingface
cond: False
x_flip: False

# HuggingFace dataset parameters
hf_dataset_name: JacobLinCool/RaindropClarity
hf_config: day  # or 'night'
hf_split: train

# Image field mapping
clean_field: clear    # background focused image (clean)
corrupt_field: drop   # raindrop-focused image (corrupt)

# Optional: limit number of images
max_imgs: 1000
```

### Field Mapping

You can customize which fields from the HuggingFace dataset are used as clean and corrupt images:

- `clean_field`: The field name in the dataset to use as the clean/target image (default: "clear")
- `corrupt_field`: The field name in the dataset to use as the corrupt/degraded image (default: "drop")

For the RaindropClarity dataset, you can use:
- `clear`: Background focused image (recommended for clean)
- `drop`: Raindrop-focused image (recommended for corrupt)
- `blur`: Raindrop-focused without raindrop (alternative option)

### Other Parameters

- `hf_dataset_name`: The name of the HuggingFace dataset
- `hf_config`: The configuration name (if the dataset has multiple configs)
- `hf_split`: The split to use (e.g., "train", "test", "validation")
- `max_imgs`: Optional limit on number of images to load
- `x_flip`: Whether to apply random horizontal flips during training
- `space_conditioning`: Whether to add spatial conditioning

## Requirements

The HuggingFace datasets library is required and should be installed automatically from requirements.txt:

```bash
pip install datasets
```

## Example: Training with RaindropClarity

```bash
# Train with day configuration
python main.py \
    dataset=raindrop_clarity_day \
    exp=custom_experiment \
    +paths.data=./datasets \
    +paths.out_path=./results
```

## Notes

- The first time you use a HuggingFace dataset, it will be downloaded and cached locally
- Large datasets may take some time to download
- You can use `max_imgs` to limit the dataset size for quick testing
- The implementation supports any HuggingFace dataset with image fields, not just RaindropClarity
