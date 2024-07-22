import torch 
import torch.nn as nn 
from encoder import * 
from libraries import * 
from decoder import * 


class transformer(nn.Module):
    def __init__(self, src_vocab_size,
                 tgt_vocab_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 embed_size: int=512,
                 num_layers: int=6,
                 ff_hidden_dim: int= 2048,
                 forward_expansion: int=4,
                 max_seq_len :int=100,
                 heads :int=8,
                 dropout :float=0.1,
                 device :str="cuda",
                 max_length :int=64):
        super(transformer, self).__init__()
        self.encoder = encoder(vocab_size=src_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               num_heads=heads,
                               ff_hidden_dim=ff_hidden_dim,
                               max_seq_length=max_seq_len,  
                               dropout=dropout,
                               )
        self.decoder = Decoder(vocab_size=tgt_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               num_heads=heads,
                               ff_hidden_dim=ff_hidden_dim,
                               max_seq_len=max_seq_len,
                               dropout=dropout)

        self.src_pad_idx = src_pad_idx 
        self.tgt_pad_idx = tgt_pad_idx 
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        print("source mask shape = ", src_mask.shape)
        return src_mask 
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        N, tgt_len = tgt.shape 
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
        padding_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        return tgt_mask.bool().to(tgt.device) & padding_mask 
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(x=src, mask=src_mask)
        # print("transformer source shape = ", src.shape, " -- target shape = ", tgt.shape)
        # print("transformer source mask  = ", src_mask.shape, " -- target mask = ", tgt_mask.shape)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return out 
        

# if __name__=="__main__":
#     tgt = torch.randint(4,5,6)
#     print("tgt = ", tgt.shape)
#     N, tgt_len = tgt.shape 
#     tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(N, 1, tgt_len, tgt_len)
#     text = "My name is Saurav"
#     tf = Transformer(text)
    
    # print(tf.attention_scores)
    