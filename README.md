# STC Transformer Project

A lightweight, from-scratch implementation of a Character-Level Transformer Language Model utilizing the **Apple MLX Framework** to leverage Apple Silicon's unified memory. 

This project trains a custom transformer specifically on a corpus of knowledge relating to swords, metallurgy, and blacksmithing, scraped directly from Wikipedia. 

## Overview

The project is structured into five distinct phases, matching the original technical roadmap (`ROADMAP.md`):

1. **Phase 1: Environment and Data Engineering** (`src/data/scraper.py`)
   - Scrapes Wikipedia articles under categories like "Swords", "Metallurgy", and "Blacksmithing".
   - Cleans the HTML and compiles the raw text into `stc_training_data.txt`.
2. **Phase 2: Tokenization** (`src/tokenizer/bpe.py`)
   - Uses a custom Character-Level Tokenizer.
   - Translates raw strings into integer arrays (tensors) that the neural network can process.
3. **Phase 3: The Transformer Architecture Build** (`src/model/transformer.py`)
   - Implements the "Attention is All You Need" architecture entirely in `mlx.nn`.
   - Includes Embedding Layers, a Multi-Headed Self-Attention mechanism, Feed-Forward Networks, and standard Transformer Blocks.
4. **Phase 4: The Training Loop** (`src/training/train.py`)
   - Prepares randomized batches from the dataset and computes Cross-Entropy Loss.
   - Leverages MLX’s lazy evaluation and automatic differentiation (`mx.compile` and `nn.value_and_grad`) to rapidly train the ~5 Million parameter model over 5000 iterations.
5. **Phase 5: Inference and Formatting** (`src/inference/generate.py`)
   - The interactive generation loop.
   - Takes a starting prompt, predicts the next character token, appends it, and repeats to construct novel text inspired by the metallurgical corpus.

---

## Setup and Installation

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4) recommended for MLX.
- Python 3.9+

**1. Activate the Virtual Environment:**
A virtual environment has already been created. To activate it, run:
```bash
source venv/bin/activate
```

**2. Install Dependencies (if not already installed):**
```bash
pip install -r requirements.txt
```

---

## Usage

A convenient `main.py` script is provided to run each phase of the project seamlessly. Make sure your virtual environment is active first.

### Step 1: Scrape the Data
This pulls the raw data from Wikipedia and saves it as `stc_training_data.txt` (~1MB of text).
```bash
python main.py scrape
```

### Step 2: Train the Model
This will initialize the MLX Transformer model and train it using the scraped text corpus. It will print the training and validation loss every 500 steps. The final model weights are saved as `model_weights.safetensors` and the vocabulary map as `tokenizer.json`.
```bash
python main.py train
```

### Step 3: Run Inference (Generate Text)
Once the model is trained, you can interact with it! This launches a CLI loop where you can input prompts.
```bash
python main.py generate
```

**Example Prompts:**
- `"The high-carbon steel "`
- `"how do I build a sword?"`
- `"iron is "`

*(Type `quit` or `exit` to stop the generation loop).*

---

## Project Structure

```text
stc/
├── main.py                     # Primary entry point
├── ROADMAP.md                  # The project plan and architecture concepts
├── requirements.txt            # Python dependencies
├── src/
│   ├── data/
│   │   └── scraper.py          # Wikipedia scraping logic
│   ├── inference/
│   │   └── generate.py         # Autoregressive generation loop
│   ├── model/
│   │   └── transformer.py      # MLX Transformer neural network code
│   ├── tokenizer/
│   │   └── bpe.py              # Tokenizer logic
│   └── training/
│       └── train.py            # Training loop, loss computation, and optimizers
```
