# Bolbosh: A Multi-Speaker Text-to-Speech System for Kashmiri

This repository contains the official code and test set for **Bolbosh**, a multi-speaker text-to-speech (TTS) system for the Kashmiri language (Persio-Arabic script). Bolbosh adapts the [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) framework — a non-autoregressive, conditional flow matching (CFM) based model — to synthesize natural-sounding Kashmiri speech from text. To the best of our knowledge, this is among the first neural TTS systems for Kashmiri, a low-resource language spoken in the Kashmir region.

## Authors

*   **[Tajamul Ashraf](https://github.com/Tajamul21)**<sup>1,3,\*</sup>
*   **[Burhaan Rasheed Zargar](https://github.com/BurhaanRasheedZargar)**<sup>3,†</sup>
*   **[Saeed Abdul Muizz](https://github.com/abdulmuizz0903)**<sup>3,†</sup>
*   **Ifrah Mushtaq**<sup>2</sup>
*   **Nazima Mehdi**<sup>2</sup>
*   **Iqra Altaf Gillani**<sup>3</sup>
*   **Aadil Amin Kak**<sup>2</sup>
*   **[Janibul Bashir](https://github.com/janibbashir)**<sup>3</sup>

<small>
<sup>1</sup> King Abdullah University of Science and Technology (KAUST) <br>
<sup>2</sup> Department of Linguistics, University of Kashmir <br>
<sup>3</sup> Gaash Lab, National Institute of Technology Srinagar <br>
<sup>*</sup> Corresponding Author <br>
<sup>†</sup> Equal Contribution <br>
</small>

**Contact:** gaash@nitsri.ac.in

---

## Highlights

- **Multi-speaker Kashmiri TTS** trained on 424 speakers by combining the Rasa dataset and IndicVoices corpus, with inference targeting two high-quality Rasa speakers (male and female).
- **Transfer learning** from a pre-trained English multi-speaker model (VCTK), fine-tuned on Kashmiri data.
- **Character-level text processing** with a custom Kashmiri text normalizer that handles Persio-Arabic script, diacritics, digit-to-word expansion, and Unicode canonicalization — no phonemizer required.
- **Conditional Flow Matching** decoder with an ODE-based inference procedure enabling fast, high-quality synthesis in as few as 10 steps.
- **HiFi-GAN vocoder** for waveform generation from predicted mel-spectrograms.

---

## Repository Structure

```
.
├── configs/                    # Hydra configuration files
│   ├── train.yaml              # Main training config
│   ├── data/                   # Dataset configs (kashmiri, ljspeech, vctk, rasa, ...)
│   ├── model/                  # Model architecture configs
│   ├── experiment/             # Experiment presets (e.g., multispeaker)
│   ├── callbacks/              # Training callback configs
│   ├── logger/                 # Logger configs (tensorboard, wandb, ...)
│   ├── trainer/                # Trainer configs (gpu, ddp, cpu, ...)
│   └── ...
├── matcha/                     # Core model package
│   ├── models/                 # Model definitions
│   │   ├── matcha_tts.py       # Main Matcha-TTS model
│   │   └── components/         # Encoder, decoder, flow matching, transformer
│   ├── text/                   # Text processing pipeline
│   │   ├── symbols.py          # Kashmiri + IPA symbol set (272 tokens)
│   │   ├── cleaners.py         # Text cleaners
│   │   └── ...
│   ├── data/                   # Data loading and processing
│   ├── hifigan/                # HiFi-GAN vocoder
│   ├── onnx/                   # ONNX export and inference utilities
│   ├── utils/                  # Utilities (audio, monotonic alignment, logging)
│   ├── train.py                # Training entry point
│   ├── cli.py                  # Command-line interface
│   └── app.py                  # Gradio demo application
├── testset_bolbosh/            # Test set (2,272 utterances with ground-truth audio)
│   ├── test.txt                # Test file list (path | speaker_id | text)
│   └── wavs/                   # Ground-truth WAV files
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks (placeholder)
├── mcd_eval.py                 # Mel Cepstral Distortion evaluation script
├── mcd_results.csv             # MCD evaluation results
├── denoise.py                  # Audio denoising utility
├── denoise2.py                 # Audio denoising utility (variant)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation (Cython build)
├── pyproject.toml              # Build system configuration
├── Makefile                    # Build and utility targets
└── LICENSE                     # MIT License
```

---

## Model Architecture

Bolbosh is built on the Matcha-TTS architecture with the following key components:

| Component | Details |
|-----------|---------|
| **Text Encoder** | 6-layer Transformer with Rotary Positional Embeddings (RoPE), 2 attention heads, 192 hidden channels |
| **Duration Predictor** | 2-layer CNN (filter size 256, kernel size 3) |
| **Decoder** | 1-D U-Net with skip connections and BasicTransformerBlocks (SnakeBeta activation) |
| **Flow Matching** | Conditional Flow Matching with Euler ODE solver, σ_min = 1e-4 |
| **Speaker Embedding** | 64-dimensional embedding for multi-speaker conditioning (424 speakers) |
| **Vocoder** | HiFi-GAN (universal v1) |
| **Vocabulary** | 272 character-level tokens (Kashmiri Persio-Arabic script, diacritics, punctuation, digits) |

### Modifications for Kashmiri

1. **No phonemizer**: Text processing operates at the character level using Kashmiri (Persio-Arabic) symbols directly, bypassing the eSpeak-based phonemizer used in the original Matcha-TTS.
2. **Custom normalizer**: A dedicated [KashmiriNormalizer](https://github.com/abdulmuizz0903/KashmiriNormalizer) handles Unicode canonicalization, diacritic preservation, Kashmiri-specific orthographic rules (e.g., Plat Ye), and digit-to-word expansion.
3. **Extended symbol set**: The vocabulary includes ~90 Kashmiri-specific characters including Arabic letters, Kashmiri extensions (ٹ, پ, چ, ڈ, ک, گ, ھ, ہ, ی, ۍ, ے, etc.), and diacritics (َ ُ ِ ٔ ٕ ٖ ٗ ٰ).
4. **Transfer learning**: The model is fine-tuned from an English VCTK multi-speaker checkpoint, adapting the embeddings and representations to Kashmiri.

---

## Dataset

The combined training dataset consists of **33,182 training**, **4,542 validation**, and **2,272 test** utterances from:

- **Rasa** — High-quality studio recordings from 2 speakers (1 male, 1 female) covering diverse categories (conversational, emotional, wiki, book narration, names).
- **IndicVoices** — A multilingual corpus contributing 422 additional Kashmiri speakers.

Audio is sampled at **22,050 Hz**. Mel-spectrograms are computed with 80 mel bands, an FFT size of 1024, hop length of 256, and a frequency range of 0–8,000 Hz.

The test set (`testset_bolbosh/`) contains 2,272 utterances with ground-truth audio and is included in this repository.

---

## Installation

### Prerequisites

- Python ≥ 3.9
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd Bolbosh

# Install dependencies
pip install -r requirements.txt

# Build the monotonic alignment Cython extension
python setup.py build_ext --inplace
```

---

## Training

Fine-tune from a pre-trained multi-speaker checkpoint:

```bash
python matcha/train.py \
    experiment=multispeaker \
    data=kashmiri \
    ckpt_path="<path-to-pretrained-checkpoint>" \
    trainer.devices=[0] \
    trainer.precision=16-mixed \
    data.num_workers=8 \
    data.batch_size=64 \
    +trainer.accumulate_grad_batches=2
```

### Data Preparation

Prepare your data in pipe-delimited format:

```
audio_path|speaker_id|text
```

Place the file lists as `data/Kashmiri/train.txt` and `data/Kashmiri/val.txt`. See `configs/data/kashmiri.yaml` for all data configuration options.

### Computing Data Statistics

Before training, compute mel-spectrogram statistics for normalization:

```bash
matcha-data-stats -i <data-config-path>
```

Update the `mel_mean` and `mel_std` values in the data configuration file accordingly.

---

## Inference

Although the model is trained on 424 speakers to learn robust Kashmiri speech representations, inference is performed using the two high-quality Rasa speakers:

| Speaker | ID | Description |
|---------|----|-------------|
| **Male** | 422 | Rasa male speaker |
| **Female** | 423 | Rasa female speaker |

### Command-Line Synthesis

```bash
# Male speaker
matcha-tts \
    --checkpoint_path <path-to-checkpoint> \
    --text "<Kashmiri text>" \
    --speaking_rate 1.0 \
    --spk 422

# Female speaker
matcha-tts \
    --checkpoint_path <path-to-checkpoint> \
    --text "<Kashmiri text>" \
    --speaking_rate 1.0 \
    --spk 423
```

### Gradio Demo

```bash
matcha-tts-app \
    --checkpoint_path <path-to-checkpoint> \
    --vocoder_path <path-to-hifigan>
```

---

## Evaluation

### Mel Cepstral Distortion (MCD)

Evaluate synthesized audio against ground-truth using the included MCD evaluation script:

```bash
python mcd_eval.py
```

This computes MCD using frequency-warped cepstral analysis (SPTK-style, 24 mel-cepstral coefficients, α = 0.55).

### Word Error Rate (WER)

WER is computed by transcribing synthesized audio using an ASR model and comparing against reference text using the `jiwer` library.

### Mean Opinion Score (MOS)

Human evaluation via listening tests on a subset of synthesized utterances.

---

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. Key configuration files:

| Config | Description |
|--------|-------------|
| `configs/data/kashmiri.yaml` | Kashmiri dataset settings (speakers, mel parameters, data paths) |
| `configs/model/matcha.yaml` | Model architecture (vocabulary size, speaker embedding, features) |
| `configs/experiment/multispeaker.yaml` | Multi-speaker experiment preset |
| `configs/train.yaml` | Top-level training configuration |

Override any parameter from the command line:

```bash
python matcha/train.py data.batch_size=32 trainer.max_epochs=500
```

---

## Test Set

The `testset_bolbosh/` directory contains the complete test set:

- **2,272 utterances** from the Rasa corpus (male speaker 422 and female speaker 423)
- Ground-truth WAV files at 22,050 Hz
- `test.txt` with entries in `audio_path|speaker_id|text` format

---

## Acknowledgements

This work builds upon the [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) framework by Mehta et al. and uses the [HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder. The Kashmiri text normalization pipeline is provided by [KashmiriNormalizer](https://github.com/abdulmuizz0903/KashmiriNormalizer).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

