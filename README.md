# Multimodal GPT — Vision-Language Model (PyTorch)

A from-scratch implementation of a **multimodal Vision-Language Model** built with PyTorch.  
The model fuses image patch tokens and text tokens into a single unified Transformer decoder stream — similar in spirit to early versions of GPT-4V and Gemini.

---

## Architecture

```
Image (224×224)
      │
 VisionEmbedding          ← Conv2d patch extraction (ViT-style, 16×16 patches)
      │  196 tokens
      ▼
  [concat]  ◄─── TokenEmbedding (text)
      │  196 + T tokens
 PositionEmbedding
      │
 TransformerBlock × N     ← Pre-LN, Causal MHA + FFN
      │
   LM Head
      │
   Logits [B, 196+T, vocab_size]
```

| Component | Description |
|---|---|
| `VisionEmbedding` | Splits image into 196 patches via `Conv2d`, outputs `[B, 196, d_model]` |
| `MultiHeadAttention` | Causal scaled dot-product attention with causal masking |
| `TransformerBlock` | Pre-LayerNorm decoder block with residual connections |
| `MultimodalGPT` | Unifies both modalities; supports training and auto-regressive generation |

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/<your-username>/multimodal-gpt.git
cd multimodal-gpt

# Install dependencies
pip install torch

# Run architecture verification
python multimodal_gpt.py
```

Expected output:
```

Forward Pass

  Input  images shape : torch.Size([2, 3, 224, 224])
  Input  text   shape : torch.Size([2, 10])
  Output logits shape : torch.Size([2, 206, 1000])
[SUCCESS] Multimodal forward pass OK!


Auto-Regressive Generation

  Generated token IDs : [[5, ...]]
  Total tokens        : 16
[SUCCESS] Image-conditioned generation OK!
```

---

## Usage

```python
from multimodal_gpt import MultimodalGPT
import torch

model = MultimodalGPT(
    vocab_size=1000,
    d_model=256,
    num_heads=4,
    num_blocks=3,
    max_text_len=50
)

# Training forward pass
images   = torch.randn(2, 3, 224, 224)
text_ids = torch.randint(0, 1000, (2, 10))
logits   = model(images, text_ids)          # [2, 206, 1000]

# Auto-regressive generation
start_token = torch.tensor([[5]])           # <START> token
generated   = model.generate(images[:1], start_token, max_new_tokens=20)
```

---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.0

---

## Author

Built as part of a self-directed Deep Learning curriculum.  
Inspired by Vision Transformer (ViT) and GPT decoder architectures.
