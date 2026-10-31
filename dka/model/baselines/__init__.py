"""
Baseline attention for comparison with RFA and DKA: dot-product multi-head attention with RoPE, SC-RoPE, xPos or no
positional encoding, optionally ALiBi, a coupled linear bias, KERPLE-log, post-softmax decay, or softmax plus one.
Presets B1-B9 in dka/training/presets.py.
"""

from .attention import MultiHeadAttentionLayer
from .pos_encoding import RoPE, SCRoPE, XPos, get_alibi_slopes
