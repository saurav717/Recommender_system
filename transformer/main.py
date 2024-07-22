from transformer import transformer as tf 
from libraries import * 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    src_vocab_size=100
    tgt_vocab_size=100 
    src_pad_idx=0
    tgt_pad_idx=0
    embed_size = 10 
    num_layers=2 
    ff_hidden_dim=20
    num_heads=6
    dropout=0.1 
    max_seq_len=50 
    batch_size=32 
    num_epochs=5 
    learning_rate=3e-4 
    
    model = tf(src_vocab_size=src_vocab_size, 
               tgt_vocab_size=tgt_vocab_size,
               src_pad_idx=src_pad_idx,
               tgt_pad_idx=tgt_pad_idx,
               embed_size=embed_size,
               num_layers=num_layers,
               ff_hidden_dim=ff_hidden_dim,
               heads=num_heads,
               dropout=dropout,
               max_seq_len=max_seq_len
               ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def get_batch(src_vocab_size,
                  tgt_vocab_size,
                  batch_size,
                  max_seq_len,
                  pad_idx):
        # src = torch.randint(1, src_vocab_size, (batch_size, max_seq_len))
        # tgt = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_len))
        src_lengths = torch.randint(2, max_seq_len, (batch_size, ))
        tgt_lengths = torch.randint(2, max_seq_len, (batch_size, ))
        
        src = torch.full((batch_size, max_seq_len), pad_idx)
        tgt = torch.full((batch_size, max_seq_len), pad_idx)
        
        for i in range(batch_size):
            src[i, :src_lengths[i]] = torch.randint(1, src_vocab_size, (src_lengths[i], ))
            tgt[i, :tgt_lengths[i]] = torch.randint(1, tgt_vocab_size, (tgt_lengths[i], ))
        
        return src.to(device), tgt.to(device)

    model.train() 
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        for batch in range(100):
            src, tgt = get_batch(src_vocab_size=src_vocab_size, 
                                 tgt_vocab_size=tgt_vocab_size,
                                 batch_size=batch_size,
                                 max_seq_len=max_seq_len,
                                 pad_idx=src_pad_idx)
            
            print("src shape = ", src.shape)
            print("tgt shape = ", tgt.shape)
            
            output = model(src, tgt[:, :-1])
            output = output.reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step() 
            
            if batch%10==0:
                print(f"Batch [{batch+1}/100], Loss: {loss.item():.4f}")
    
    print("Training Completed!")
    
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model Saved!")

if __name__=="__main__":
    main()
    