from dataclasses import dataclass

@dataclass

class MiniMindConfig:
    vocab_size: int = 6400  # 使用前确认与 tokenizer 词表大小一致
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 384
    max_seq_len: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.hidden_size//self.num_attention_heads