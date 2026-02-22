# Transformer Pretraining

Pretrain a small transformer model on text data from scratch using PyTorch.

## Structure
```
├── pretrain_transformer.ipynb
├── config.yaml
├── data/
│   └── train.txt
├── models/
│   └── checkpoints/
└── modules/
    ├── transformer.py
    ├── trainer.py
    └── data_utils.py
```

## Architecture
- Multi-head self-attention (8 heads)
- 4 transformer blocks with feed-forward networks
- Causal masking for language modeling (predicts next token)
- ~5M parameters, trainable on CPU/GPU

## Dataset
Raw text data (`train.txt`) tokenized and batched for training. Model learns causal language modeling—predicting the next token given previous ones.

## Setup
- Python 3.8+
- PyTorch 2.0+
- NumPy

## Usage
1. Open `pretrain_transformer.ipynb`
2. Configure hyperparameters in `config.yaml`
3. Run notebook to train

## Config
```yaml
Model:
  vocab_size: 5000
  d_model: 256
  n_heads: 8
  n_layers: 4
  d_ff: 1024
  max_seq_len: 128
  dropout: 0.1

Training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 1000
```
