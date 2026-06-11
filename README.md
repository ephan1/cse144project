# cse144project

## Overview

This project implements image classification using transfer learning with a pre-trained Vision Transformer (ViT-B/16) model from TorchVision.

The model is fine-tuned on the UCSC CSE 144 Spring 2026 Final Project dataset containing 100 classes and approximately 1,079 labeled training images. The final model generates predictions for the unlabeled test images and outputs a Kaggle submission file.

---

## Environment

Python version:

```
Python 3.10+
```

Required packages:

```
torch
torchvision
numpy
pandas
matplotlib
tqdm
kagglehub
Pillow
scikit-learn
```

Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm kagglehub pillow scikit-learn
```

---

## Dataset

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.competition_download(
    "ucsc-cse-144-spring-2026-final-project"
)

print(path)
```

The dataset directory should have the following structure:

```
ucsc-cse-144-spring-2026-final-project/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 99/
├── test/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 1035.jpg
└── sample_submission.csv
```

---

## Model Architecture

The model uses the pre-trained Vision Transformer:

```
ViT-B/16 (IMAGENET1K_V1)
```

Transfer learning strategy:

* All ViT parameters are initially frozen.
* The final three transformer encoder blocks are unfrozen.
* The classification head is replaced with:

```
Linear(768 → 256)
ReLU
Dropout(0.5)
Linear(256 → 100)
```

---

## Data Preprocessing

Training transforms:

* RandomResizedCrop(224)
* RandomHorizontalFlip()
* AutoAugment (ImageNet policy)
* ToTensor()
* Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])

Validation/Test transforms:

* Resize(256)
* CenterCrop(224)
* ToTensor()
* Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225])

---

## Training

Hyperparameters:

```
Loss:
CrossEntropyLoss(label_smoothing=0.1)

Optimizer:
AdamW

Learning Rates:
1e-5

Weight decay:
1e-4

Scheduler:
ReduceLROnPlateau(
    mode="max",
    factor=0.5,
    patience=5
)

Early stopping patience:
8 epochs

Batch size:
32
```

To train the model:

```bash
Run all cells in run.ipynb from top to bottom.
```

Training will:

* Train the model.
* Evaluate on the validation set after each epoch.
* Save the best checkpoint.

Saved checkpoint:

```
checkpoints/best_vit.pt
```

The checkpoint contains:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "val_acc": ...
}
```

---

## Inference

To generate predictions on the test set:

```bash
Run all cells in Final_Project.ipynb from top to bottom.
```

Inference will:

1. Load the saved checkpoint.
2. Load all test images.
3. Predict class labels.
4. Create a submission file.

Output:

```
submission.csv
```

Submission format:

```
ID,Label
0.jpg,62
1.jpg,43
2.jpg,38
...
1035.jpg,67
```

---

## Reproducibility

The following seed is used throughout the experiments:

```python
set_seed(42)
```

This sets:

* Python random seed
* NumPy random seed
* PyTorch random seed
* CUDA random seed (if applicable)

Deterministic settings are enabled whenever possible.

---

## Results

Best Kaggle Public Leaderboard Score:

``` 
0.66363

```

using the ViT-B/16 transfer learning approach described above.

![Final Kaggle score](finalscore.png)

Best weights: 
https://drive.google.com/drive/folders/1kqrI8zWGbueRNwXYjhtSfk2ZdThmTVb6?usp=sharing  


