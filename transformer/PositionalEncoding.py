from libraries import * 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int,
                 max_seq_length: int=64):
        super(PositionalEncoding,self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: torch.Tensor, shape [batch_size, seq_len, embedding_dim]

        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

# if __name__=="__main__":
#     d_model = 512 
#     max_seq_length = 64 
    
#     pe = PositionalEncoding(10, 64)
        
        
        
        
        
        