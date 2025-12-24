# Audio Deepfake Detection (Task 2)

This project detects whether a given speech audio sample is **REAL** or **AI-GENERATED**
using classical audio features and a lightweight deep learning model.

---

## Dataset
- **Fake-or-Real (FoR) Dataset**
- Version: **for-2sec**
- Contains normalized 2-second real and synthetic speech samples
- Classes:
  - REAL (human speech)
  - FAKE (AI-generated / TTS speech)

Please download the Fake-or-Real (FoR) dataset from Kaggle:
https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset
---

## Methodology

### Pipeline
Audio → MFCC Extraction → CNN → Softmax → Real/Fake Prediction

### Features
- MFCC (40 coefficients)
- Fixed-length representation for CNN compatibility

### Model
- Lightweight Convolutional Neural Network (CNN)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## Training
- Train/Validation split: **80% / 20%**
- Batch size: 8
- Epochs: 10
- Validation accuracy printed during training

---

## Evaluation Results

The model was evaluated on a small held-out test set containing
10 real and 10 fake audio samples.

- Accuracy: 0.75
- Precision: 1.00
- Recall: 0.67
- F1 Score: 0.80

High precision indicates that all audio samples classified as fake
were indeed fake, while lower recall suggests that some fake samples
were misclassified as real. This behavior is expected for a lightweight
model evaluated on a small dataset.

### Note on EER
Equal Error Rate (EER) is a commonly used metric in audio spoof detection.
It was not implemented in this version but can be added as future work.

---

## Inference

Run inference on a single audio file:
```bash
python -m src.inference path/to/audio.wav
