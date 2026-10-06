# Attention-Based Image Captioning using ResNet18 and LSTM

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

A PyTorch implementation of an attention-based image captioning model that generates natural language descriptions for images using a pretrained ResNet18 encoder and LSTM decoder with spatial attention.

## Overview

This project implements an encoder-decoder architecture with attention for automatic image captioning on the Flickr8k dataset. The encoder extracts spatial features using pretrained ResNet18, while the decoder generates captions word-by-word using an LSTM with attention over 7×7 feature maps.

**Key Features:**
- Pretrained ResNet18 encoder with 1×1 convolution (512 → 128 channels)
- Spatial attention mechanism over 7×7 feature maps
- LSTMCell-based decoder with word embeddings
- Early stopping with patience = 5 epochs
- Greedy decoding for inference
- Checkpoint saving for best model

## Dataset

**Flickr8k Dataset:**
- 8,000 images with 5 captions each
- Images resized to 224×224 for ResNet18
- Captions tokenized with special tokens: `<start>`, `<end>`, `<pad>`, `<unk>`
- Vocabulary built from training captions with minimum frequency threshold

## Model Architecture

```
Input Image (224×224×3)
        ↓
  ResNet18 Encoder (pretrained)
        ↓
  Feature Maps (512×7×7)
        ↓
  1×1 Conv (512 → 128)
        ↓
  Spatial Attention (7×7)
        ↓
  Context Vector (128)
        ↓
  LSTM Decoder + Embeddings
        ↓
  Generated Caption
```

### Encoder
- Pretrained ResNet18 (ImageNet weights)
- Final FC and pooling layers removed
- 1×1 convolution reduces channels from 512 to 128
- Outputs spatial features: `128 × 7 × 7`

### Attention Mechanism
- Bahdanau-style additive attention
- Computes attention weights over 49 spatial locations (7×7)
- Generates context vector as weighted sum of encoder features
- Attention weights dynamically focus on relevant image regions

### Decoder
- Word embedding layer
- LSTMCell for sequential generation
- Concatenates embedded word with attention context
- Linear projection to vocabulary size
- Greedy decoding: selects word with highest probability

## Training

**Loss Function:** CrossEntropyLoss (ignores padding index)

**Optimizer:** Adam (lr=0.001)

**Early Stopping:** Patience = 5 epochs (monitors validation loss)

**Checkpoint:** Best model saved as `best_model.pth`

## Project Structure

```
.
├── train.py          # Training script
├── model.py          # Encoder, Decoder, Attention modules
├── dataset.py        # Custom Dataset and DataLoader
├── utils.py          # Vocabulary builder and utilities
├── infer.py          # Inference script
├── captions.txt      # Caption file (Flickr8k format)
└── best_model.pth    # Trained model checkpoint
```

## Installation

```bash
# Clone repository
git clone https://github.com/2300040099/image-captioning-resnet-lstm.git
cd image-captioning-resnet-lstm

# Install dependencies
pip install torch torchvision pillow numpy
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow
- NumPy

## Usage

### Training

```python
python train.py
```

The script will:
1. Load and preprocess Flickr8k dataset
2. Build vocabulary from captions
3. Train encoder-decoder model with attention
4. Save best model checkpoint as `best_model.pth`
5. Stop early if validation loss doesn't improve for 5 epochs

### Inference

```python
python infer.py
```

Generates captions for test images using greedy decoding. The script loads `best_model.pth` and produces captions word-by-word until `<end>` token or max length.

**Example Output:**
```
Image: sample.jpg
Caption: a dog is running through the grass with a ball
```

## Model Details

**Hyperparameters:**
- Embedding dimension: 256
- LSTM hidden size: 256
- Encoder output dimension: 128
- Learning rate: 0.001
- Batch size: 32

**Training Process:**
1. Encoder extracts spatial features from images
2. Attention computes context vector at each time step
3. Decoder generates next word conditioned on context and previous word
4. Loss computed against ground truth captions
5. Early stopping prevents overfitting

## Future Improvements

- Implement beam search decoding for better caption quality
- Add BLEU score evaluation
- Experiment with larger encoders (ResNet50, EfficientNet)
- Try Transformer-based decoder
- Add attention visualization

## References

- **Show, Attend and Tell:** Xu et al. (2015) - [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
- **Deep Residual Learning:** He et al. (2015) - [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## License

MIT License

## Author

**Your Name**
- GitHub: [@2300040099](https://github.com/2300040099)
- Email: sshruthi2404@gmail.com

---

**⭐ Star this repo if you find it helpful!**
