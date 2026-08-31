import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """PyTorch implementation of LoRA (Low-Rank Adaptation) Layer."""
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Base linear layer (Frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA trainable matrices A and B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # Initialize LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return base_out + lora_out

class GemmaLoRAModel(nn.Module):
    """Gemma-style Transformer LM with LoRA Adapters on Attention Projections."""
    def __init__(self, vocab_size=32000, hidden_dim=768, num_layers=4, num_heads=12, r=8):
        super(GemmaLoRAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": LoRALinear(hidden_dim, hidden_dim, r=r),
                "v_proj": LoRALinear(hidden_dim, hidden_dim, r=r),
                "k_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "out_proj": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "mlp": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            # Self Attention with LoRA on Q and V
            norm_x = layer["norm1"](x)
            q = layer["q_proj"](norm_x)
            k = layer["k_proj"](norm_x)
            v = layer["v_proj"](norm_x)
            
            # Simple scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            attn_probs = torch.softmax(attn_weights, dim=-1)
            attn_out = torch.matmul(attn_probs, v)
            attn_out = layer["out_proj"](attn_out)
            
            x = x + attn_out
            x = x + layer["mlp"](layer["norm2"](x))
            
        logits = self.lm_head(x)
        return logits

    def mark_only_lora_as_trainable(self):
        """Freeze base model weights and unfreeze ONLY LoRA parameters (lora_A, lora_B)."""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
                
        print(f"🔒 Model Parameters Frozen! Total Params: {total_params:,} | Trainable LoRA Params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    def get_lora_state_dict(self):
        """Returns ONLY LoRA adapter weights for lightweight backup."""
        return {k: v.cpu() for k, v in self.state_dict().items() if "lora_A" in k or "lora_B" in k}
