"""
Multimodal GPT — Vision-Language Model
=======================================
A from-scratch PyTorch implementation of a multimodal architecture
that fuses image patch tokens with text tokens in a single unified
Transformer decoder stream.

Architecture Overview:
  - VisionEmbedding   : Splits image into patches (ViT-style) via Conv2d
  - MultiHeadAttention: Causal scaled dot-product attention
  - TransformerBlock  : Pre-LN decoder block (MHA + FFN + residuals)
  - MultimodalGPT     : Concatenates visual + text tokens → Transformer → logits
                        Also supports auto-regressive text generation from images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------

class VisionEmbedding(nn.Module):
    """
    Projects image patches into a continuous visual token space (Vision Transformer approach).
    Takes an image and cuts it into non-overlapping grids/patches.
    """

    def __init__(self, in_channels=3, patch_size=16, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        # Using Conv2d as a highly efficient patch extraction trick
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: [Batch_Size, Channels, Height, Width]  e.g., [2, 3, 224, 224]
        x = self.proj(x)                        # [B, d_model, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)        # [B, Number_of_Patches, d_model]
        return x


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Attention mechanism.
    Allows tokens to dynamically focus on relevant context while masking future text tokens.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        # Combined projections for training efficiency
        self.q_linear   = nn.Linear(d_model, d_model, bias=False)
        self.k_linear   = nn.Linear(d_model, d_model, bias=False)
        self.v_linear   = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        # Project and reshape for multi-head computation: [B, num_heads, T, head_dim]
        q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Causal mask — text tokens cannot look ahead
        mask   = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        # Merge heads back
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_linear(context)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard Decoder Transformer Block combining:
      - Multi-Head Attention
      - Feed-Forward Network (4× expansion)
      - Layer Normalization with Residual Connections  (Pre-LN design)
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))   # Attention  sub-layer
        x = x + self.ffn(self.ln2(x))   # FFN        sub-layer
        return x


# ---------------------------------------------------------------------------
# Multimodal GPT
# ---------------------------------------------------------------------------

class MultimodalGPT(nn.Module):
    """
    Unified Vision-Language Model.

    Concatenates visual patch tokens and textual word tokens into a single
    unified context stream and passes them through a Transformer decoder.

    Supports:
      - forward()  : teacher-forced training pass → returns logits
      - generate() : auto-regressive image-conditioned text generation
    """

    def __init__(self, vocab_size, d_model, num_heads, num_blocks, max_text_len):
        super().__init__()
        self.max_text_len = max_text_len
        self.num_patches  = 196  # (224 / 16) ** 2 = 196 patches from a 224×224 image

        self.vision_encoder    = VisionEmbedding(d_model=d_model)
        self.token_embedding   = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_text_len + self.num_patches, d_model)

        self.blocks = nn.Sequential(*[TransformerBlock(d_model, num_heads) for _ in range(num_blocks)])
        self.ln_f   = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, images, text_idx):
        # 1. Image → visual tokens   [B, 196, d_model]
        img_features  = self.vision_encoder(images)

        # 2. Text  → textual tokens  [B, T, d_model]
        text_features = self.token_embedding(text_idx)

        # 3. Multimodal Fusion: concatenate along the sequence dimension
        combined = torch.cat((img_features, text_features), dim=1)  # [B, 196+T, d_model]

        # 4. Add positional information
        B, Total_T, _ = combined.shape
        pos_emb = self.position_embedding(torch.arange(Total_T, device=text_idx.device))
        x = combined + pos_emb

        # 5. Transformer stack
        x = self.blocks(x)
        x = self.ln_f(x)

        # 6. Vocabulary logits
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, images, start_token_idx, max_new_tokens):
        """
        Auto-regressive image-conditioned text generation.

        Args:
            images          : Tensor [B, 3, H, W]
            start_token_idx : Tensor [B, 1] — e.g., <START> token index
            max_new_tokens  : int — number of tokens to generate

        Returns:
            Tensor [B, 1 + max_new_tokens] — generated token IDs
        """
        text_idx = start_token_idx

        for _ in range(max_new_tokens):
            # Crop context to allowed maximum length
            text_cond = text_idx[:, -self.max_text_len:]

            logits          = self(images, text_cond)
            next_tok_logits = logits[:, -1, :]                          # last position
            probs           = F.softmax(next_tok_logits, dim=-1)
            next_token      = torch.multinomial(probs, num_samples=1)   # sample

            text_idx = torch.cat((text_idx, next_token), dim=1)

        return text_idx


# ---------------------------------------------------------------------------
# Quick Architecture Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE   = 1000
    D_MODEL      = 256
    NUM_HEADS    = 4
    NUM_BLOCKS   = 3
    MAX_TEXT_LEN = 50

    model = MultimodalGPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_text_len=MAX_TEXT_LEN
    )

    # --- Forward pass test ---
    images    = torch.randn(2, 3, 224, 224)                    # 2 RGB images
    text_ids  = torch.randint(0, VOCAB_SIZE, (2, 10))          # 2 sequences of length 10
    logits    = model(images, text_ids)

    print("=" * 55)
    print("Forward Pass")
    print("=" * 55)
    print(f"  Input  images shape : {images.shape}")
    print(f"  Input  text   shape : {text_ids.shape}")
    print(f"  Output logits shape : {logits.shape}")
    print("[SUCCESS] Multimodal forward pass OK!\n")

    # --- Generation test ---
    start_token    = torch.tensor([[5]])                        # <START> token
    single_image   = torch.randn(1, 3, 224, 224)
    generated      = model.generate(single_image, start_token, max_new_tokens=15)

    print("=" * 55)
    print("Auto-Regressive Generation")
    print("=" * 55)
    print(f"  Generated token IDs : {generated.tolist()}")
    print(f"  Total tokens        : {generated.shape[1]}")
    print("[SUCCESS] Image-conditioned generation OK!")
