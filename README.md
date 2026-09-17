# Kokoro-7M 2-Voice Training on Google Colab (T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ha466/kokoro_nano/blob/main/colab_training.ipynb)

This repository contains everything needed to train the 7.48M Kokoro student with **1 Female (`af_bella`) and 1 Male (`am_adam`)** voice on a free Google Colab T4 GPU.

---

## Quick Start on Google Colab

### Option 1: Open Directly in Colab (Recommended)

1. Click the **[Open In Colab](https://colab.research.google.com/github/ha466/kokoro_nano/blob/main/colab_training.ipynb)** badge above.
2. Set runtime to GPU: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU**.
3. Run the cells in order:
   * **Step 0**: Clone repository (`git clone https://github.com/ha466/kokoro_nano.git`).
   * **Step 1**: Verify GPU.
   * **Step 2**: Install requirements (`pip install -r requirements.txt`).
   * **Step 3**: Generate teacher dataset (`generate_teacher_dataset.py`).
   * **Step 4**: Run distillation training (`train_student.py`).
   * **Step 5**: Test speech generation and listen to both voices.
   * **Step 6**: Download the trained checkpoint bundle.

---

### Option 2: Clone from a Blank Colab Notebook

In a fresh Colab notebook with T4 GPU enabled, simply run:

```bash
!git clone https://github.com/ha466/kokoro_nano.git
%cd kokoro_nano
!bash run_colab.sh
```

---

## Contents of this Repository

* **`colab_training.ipynb`**: Interactive notebook with visual audio playback and automated git clone.
* **`requirements.txt`**: Pinned Python dependencies for the Colab environment.
* **`run_colab.sh`**: Automated shell execution script.
* **`training/generate_teacher_dataset.py`**: Paired 2-voice teacher dataset generator.
* **`training/train_student.py`**: Distillation trainer with 2-voice style conditioning.
* **`training/styletts2_gan/`**: Multi-Period & Multi-Resolution GAN discriminators and loss functions.
* **`kokoro_patched/`**: Vendored Kokoro model architecture supporting custom decoder dimensions.
* **`config.json`**: Student architecture hyperparameter configuration.
* **`load_model.py`**: Safe model loader for inference.
* **`kokoro_en_7m.pth`**: Pre-trained baseline weights for warm start.
