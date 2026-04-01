# Unsupervised Neural Network for Multi-Genre Music Generation
**CSE425 / EEE474 — Spring 2026**

This project builds four progressively complex unsupervised generative models for MIDI music.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets

#### MAESTRO (required)
Download from https://magenta.tensorflow.org/datasets/maestro and extract to `data/raw_midi/maestro/`.

#### Groove MIDI (optional for Task 2 multi-genre)
Download from https://magenta.tensorflow.org/datasets/groove and extract to `data/raw_midi/groove/`.

#### Lakh MIDI (optional for Task 2 multi-genre)
Download Lakh MIDI and extract cleaned MIDI files to `data/raw_midi/lakh/`.

### 3. Run preprocessing
```bash
# MAESTRO only
python src/preprocessing/midi_parser.py --dataset maestro

# Groove only
python src/preprocessing/midi_parser.py --dataset groove

# Lakh only
python src/preprocessing/midi_parser.py --dataset lakh

# Process every supported dataset found under data/raw_midi/
python src/preprocessing/midi_parser.py --dataset all
```
This writes split files like `{genre}_train.npy`, `{genre}_validation.npy`, `{genre}_test.npy` to `data/processed/`.  
Alternatively, open `notebooks/preprocessing.ipynb` for a step-by-step walkthrough.

---

## Training

```bash
# Task 1 — LSTM Autoencoder
python src/training/train_ae.py

# Task 2 — VAE (single genre)
python src/training/train_vae.py --genres maestro --epochs 50

# Task 2 — VAE (multi-genre)
python src/training/train_vae.py --genres maestro,groove,lakh --epochs 50

# Task 3 — Transformer (single or multi-genre)
python src/training/train_transformer.py --genres maestro --epochs 50
python src/training/train_transformer.py --genres maestro,groove,lakh --epochs 50

# Task 4 — RLHF
python src/training/train_rlhf.py --rl_steps 30 --episodes_per_step 8 --genre maestro
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

# Compare all models (baselines + Tasks 1-4)
python src/evaluation/metrics.py --all

# Compare RLHF before-vs-after outputs
python src/evaluation/metrics.py --compare_rlhf
```

Computes pitch histogram similarity, rhythm diversity, and repetition ratio for generated MIDI files and saves reports including:
- `outputs/generated_midis/evaluation_results.csv`
- `outputs/generated_midis/all_models_comparison.csv`
- `outputs/survey_results/task4_comparison.csv`

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
│   ├── models/             # autoencoder, vae, transformer, diffusion (placeholder)
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

---

## Notes

- `src/models/diffusion.py` is intentionally a placeholder and not part of the core four required tasks.
