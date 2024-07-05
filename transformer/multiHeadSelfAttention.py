from libraries import *  

class multiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dimensions: dict):
        super(multiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size 
        self.num_heads = heads 
        self.dimensions = dimensions

        # self.head_dim = embed_size//heads        
        # assert (self.head_dim*heads == self.embed_size), "Embed size needs to be  divisible by heads"
        
        self.keys = nn.Linear(embed_size, dimensions["keys"]*heads)
        self.queries = nn.Linear(embed_size, dimensions["queries"]*heads)
        self.values = nn.Linear(embed_size, dimensions["values"]*heads)
        self.fc_out = nn.Linear(heads*dimensions["values"], embed_size)
        
        self.attention = None
        
    
    def forward(self, embedding: torch.Tensor,
                      mask   : torch.Tensor = None): 
        N = embedding.shape[0] # Number of inputs
        seq_len = embedding.shape[1] # Sequence lengths
        
        # Projections 
        values = self.values(embedding).view(N, seq_len, self.num_heads, self.dimensions["values"])
        keys = self.keys(embedding).view(N, seq_len, self.num_heads, self.dimensions["keys"])
        queries = self.queries(embedding).view(N, seq_len, self.num_heads, self.dimensions["queries"])
        
        # Attention 
        compatibility = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            compatibility = compatibility.masked_fill(mask==0, float("-1e20"))
        attention = torch.softmax(compatibility / math.sqrt(self.dimensions["keys"]), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        self.attention = out
        
        # Concatenate heads and project 
        out = out.reshape(N, seq_len, self.num_heads*self.dimensions["values"])
        out = self.fc_out(out)
        return out 

if __name__ == "__main__":
    attn = multiHeadSelfAttention(embed_size=10,
                                  heads=5,
                                  dimensions={"keys": 5, "queries": 5, "values": 7})
    
    embeddings = torch.randn(2, 10, 10)
    x = attn(embeddings)
    
    
    
    
        
        