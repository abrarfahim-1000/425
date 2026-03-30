# Unsupervised Neural Network for Multi-Genre Music Generation
**CSE425 / EEE474 — Spring 2026**

This project builds four progressively complex unsupervised generative models for MIDI music using the MAESTRO dataset.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download MAESTRO dataset
Download from https://magenta.tensorflow.org/datasets/maestro and extract to `data/raw_midi/maestro/`.

### 3. Run preprocessing
```bash
python src/preprocessing/midi_parser.py
```
This writes `maestro_train.npy`, `maestro_validation.npy`, `maestro_test.npy` to `data/processed/`.  
Alternatively, open `notebooks/preprocessing.ipynb` for a step-by-step walkthrough.

---

## Training

```bash
# Task 1 — LSTM Autoencoder
python src/training/train_ae.py

# Task 2 — VAE
python src/training/train_vae.py --genres maestro --epochs 50

# Task 3 — Transformer
python src/training/train_transformer.py --genres maestro --epochs 50
```

Checkpoints are saved to `checkpoints/`. Training curves are saved to `outputs/plots/`.

---

## Generation

```bash
# Sample from AE / VAE latent space
python src/generation/sample_latent.py --model ae --num_samples 5
python src/generation/sample_latent.py --model vae --num_samples 8

# Generate with Transformer
python src/generation/generate_music.py --model transformer --num_samples 10

# Baselines
python src/generation/generate_music.py --model baseline_random
python src/generation/generate_music.py --model baseline_markov
```

Generated MIDI files are saved to `outputs/generated_midis/`.

---

## Evaluation

```bash
python src/evaluation/metrics.py
```

Computes pitch histogram similarity, rhythm diversity, and repetition ratio for all generated MIDI files and saves a CSV report to `outputs/generated_midis/evaluation_results.csv`.

---

## Project Structure

```
├── data/
│   ├── raw_midi/           # Raw MIDI files (MAESTRO etc.)
│   └── processed/          # Pre-processed .npy segments
├── notebooks/
│   ├── preprocessing.ipynb # Preprocessing pipeline walkthrough
│   └── baseline_markov.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing/      # midi_parser, piano_roll, tokenizer
│   ├── models/             # autoencoder, vae, transformer
│   ├── training/           # train_ae, train_vae, train_transformer
│   ├── evaluation/         # metrics, pitch_histogram, rhythm_score
│   └── generation/         # sample_latent, generate_music, midi_export
├── outputs/
│   ├── generated_midis/
│   └── plots/
└── checkpoints/
```

---

## Hardware Note

Tested on AMD Ryzen 5 7500F + Intel Arc B580 (12 GB VRAM). PyTorch 2.x supports the Arc GPU natively via the `xpu` backend. The device is auto-detected in `src/config.py`.
