from libraries import * 
from multiHeadSelfAttention import multiHeadSelfAttention
from PositionalEncoding import PositionalEncoding

class DecoderBlock(nn.Module):
    def __init__(self, max_seq_len:int,
                 embed_size:int, 
                 num_heads: int,
                 ff_hidden_dim: int,
                 dropout: float = 0.1
                 ):
        super(DecoderBlock, self).__init__()
        
        # Masked Self-Attention
        self.self_attention = multiHeadSelfAttention(embed_size, heads=num_heads, 
                                                     dimensions={"keys": 5, "queries": 5, "values": 7})
        self.norm1 = nn.LayerNorm(embed_size)
        
        self.cross_attention = multiHeadSelfAttention(embed_size, heads=num_heads,
                                                      dimensions={"keys": 5, "queries": 5, "values": 7}) 
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor=None,
                tgt_mask: torch.Tensor=None,
                ) -> torch.Tensor:
        attn_output = self.self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        cross_attn_output = self.cross_attention(x, keys=encoder_output, values=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = x + self.dropout(x)
        return x 

        
class Decoder(nn.Module):
    def __init__(self, vocab_size: int,
                 embed_size: int,
                 num_layers: int,
                 num_heads: int,
                 ff_hidden_dim: int,
                 max_seq_len: int,
                 dropout: float=0.1):
        super(Decoder, self).__init__()
        self.embed_size = embed_size 
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = PositionalEncoding(embed_size, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ]) 
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                encoder_output: torch.Tensor,
                src_mask: torch.Tensor=None,
                tgt_mask: torch.Tensor=None
                ):
        N, seq_len = x.shape 
        
        x = self.word_embedding(x) * math.sqrt(self.embed_size)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        output = self.fc_out(x)
        return output 
        
        
        
        
        