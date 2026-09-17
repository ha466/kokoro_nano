# Kokoro-7M 2-Voice Training on Google Colab (T4 GPU)

This package contains everything needed to train the 7.48M Kokoro student with **1 Female (`af_bella`) and 1 Male (`am_adam`)** voice on a free Google Colab T4 GPU.

---

## Quick Start on Google Colab

### Method 1: Using the Jupyter Notebook (`colab_training.ipynb`)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** $\rightarrow$ **Upload notebook** and upload [`colab_training.ipynb`](./colab_training.ipynb).
3. Set your runtime to GPU:
   * Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU**.
4. Upload this `cloud` folder to your Colab session:
   * Zip the `cloud` folder on your PC (`cloud.zip`).
   * In the Colab file browser on the left, upload `cloud.zip`.
   * Run in a cell: `!unzip -q cloud.zip && cp -r cloud/* .`
5. Run the cells in order:
   * **Cell 1**: Verify GPU.
   * **Cell 2**: Install requirements (`pip install -r requirements.txt`).
   * **Cell 3**: Generate teacher dataset (`generate_teacher_dataset.py`).
   * **Cell 4**: Run training (`train_student.py`).
   * **Cell 5**: Listen to generated audio from both female and male voices.
   * **Cell 6**: Download the trained `student.pth` checkpoint.

---

### Method 2: One-Command Execution via Terminal / Script

If running in a Colab code cell:
```bash
!bash run_colab.sh
```

---

## Contents of this `cloud/` Package

* **`colab_training.ipynb`**: Interactive notebook with visual audio playback.
* **`requirements.txt`**: Pinned Python dependencies for Colab environment.
* **`run_colab.sh`**: Automated shell execution script.
* **`training/generate_teacher_dataset.py`**: Paired 2-voice teacher dataset generator.
* **`training/train_student.py`**: Distillation trainer with 2-voice support.
* **`training/styletts2_gan/`**: Multi-Period & Multi-Resolution GAN discriminators and loss functions.
* **`kokoro_patched/`**: Vendored Kokoro model architecture supporting custom decoder dimensions.
* **`config.json`**: Student architecture hyperparameter configuration.
* **`load_model.py`**: Safe model loader for inference.
* **`kokoro_en_7m.pth`**: Pre-trained weights for optional warm start.
