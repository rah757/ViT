import math
import torch
from torch import nn

class NewGELUActivation(nn.Module):
    """
    GELU activation function as used in BERT and GPT.
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
        ))

class PatchEmbeddings(nn.Module):
    """
    Convert image into patches and project to a vector space.
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size, 
            kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, hidden_size, num_patches_h, num_patches_w)
        x = self.projection(x)
        # (B, hidden_size, num_patches_h, num_patches_w) -> (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    """
    Combine patch embeddings with a CLS token and positional embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    """
    A single attention head.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, value)
        return (out, probs)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module using multiple AttentionHead instances.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.qkv_bias = config["qkv_bias"]

        self.heads = nn.ModuleList([
            AttentionHead(self.hidden_size, self.attention_head_size, config["attention_probs_dropout_prob"], self.qkv_bias)
            for _ in range(self.num_attention_heads)
        ])

        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([o[0] for o in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if output_attentions:
            attention_probs = torch.stack([o[1] for o in attention_outputs], dim=1)
            return (attention_output, attention_probs)
        else:
            return (attention_output, None)

class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention optimized to handle all heads simultaneously.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.qkv_bias = config["qkv_bias"]

        self.qkv_projection = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        B, N, _ = x.size()
        qkv = self.qkv_projection(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(B, N, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(B, N, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_attention_heads * self.attention_head_size)
        out = self.output_projection(out)
        out = self.output_dropout(out)

        if output_attentions:
            return (out, probs)
        else:
            return (out, None)

class MLP(nn.Module):
    """
    Feed-forward network (MLP).
    """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer encoder block.
    """
    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        self.attention = FasterMultiHeadAttention(config) if self.use_faster_attention else MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        attn_out, attn_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attn_out
        mlp_out = self.mlp(self.layernorm_2(x))
        x = x + mlp_out
        return (x, attn_probs if output_attentions else None)

class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple Block layers.
    """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(probs)
        return (x, all_attentions if output_attentions else None)

class ViTForClassfication(nn.Module):
    """
    Vision Transformer for classification tasks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        x = self.embedding(x)
        x, attentions = self.encoder(x, output_attentions=output_attentions)
        logits = self.classifier(x[:, 0, :])
        return (logits, attentions if output_attentions else None)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif hasattr(module, "position_embeddings") and hasattr(module, "cls_token"):
            nn.init.trunc_normal_(
                module.position_embeddings.data, std=self.config["initializer_range"]
            )
            nn.init.trunc_normal_(
                module.cls_token.data, std=self.config["initializer_range"]
            )
