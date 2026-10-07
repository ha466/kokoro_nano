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
   * **Step 2**: Create the locked `uv` environment (`uv sync`).
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
* **`pyproject.toml`** and **`uv.lock`**: Canonical, locked `uv` project dependencies.
* **`requirements.txt`**: Compatibility dependency list for tools that cannot use `uv`.
* **`run_colab.sh`**: Automated shell execution script.
* **`training/generate_teacher_dataset.py`**: Paired 2-voice teacher dataset generator.
* **`training/train_student.py`**: Distillation trainer with 2-voice style conditioning.
* **`training/styletts2_gan/`**: Multi-Period & Multi-Resolution GAN discriminators and loss functions.
* **`kokoro_patched/`**: Vendored Kokoro model architecture supporting custom decoder dimensions.
* **`config.json`**: Student architecture hyperparameter configuration.
* **`load_model.py`**: Safe model loader for inference.
* **`kokoro_en_7m.pth`**: Pre-trained baseline weights for warm start.

## Faster Dataset Generation

The teacher generator supports batched synthesis. On a Colab T4, start with
four clips per batch and lower the value if you encounter an out-of-memory
error:

```bash
uv run python training/generate_teacher_dataset.py \
  --texts sentences.txt \
  --female-voice af_bella \
  --male-voice am_adam \
  --batch-size 4 \
  --out-dir dist_twovoice
```

Each batch is right-padded only for inference; `index.jsonl` and
`audio.i16.bin` still contain the original, individually trimmed clips.

## Fast dependency setup with uv

`uv` resolves and installs the project dependencies in parallel. `pyproject.toml`
and `uv.lock` are the source of truth; every project command should be launched
through `uv run`:

```bash
python -m pip install --quiet --upgrade uv
uv sync
```

The English spaCy model is included as a project dependency, so do not run
`spacy download` separately; `uv sync` installs it into `.venv`.

---

## Credits & Acknowledgements

* **Base Distillation & Architecture**: [`oddadmix/Kokoro-7M-Distill`](https://huggingface.co/oddadmix/Kokoro-7M-Distill) by **oddadmix** for the ultra-compact 7.48M student architecture, training configurations, and pre-trained weights.
* **Original Teacher Model**: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) by **hexgrad** for the high-quality 82M open-weight TTS model.
* **Generative Architecture**: Based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) (Yinghao Aaron Li et al.).

