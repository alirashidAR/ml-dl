import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_len,d_model=512):
        super(PositionalEncoding,self).__init__()
        self.d_model = d_model
        pos = torch.arange(0, max_seq_len,dtype = torch.float).unsqueeze(1)
        
        frequency = torch.pow(10000,-torch.arange(0,d_model,2,dtype = torch.float)/self.d_model)
        pe = torch.zeros((max_seq_len,d_model))
        pe[:,0::2] = torch.sin(pos * frequency)
        pe[:,1::2] = torch.cos(pos * frequency)
    
        self.register_buffer('pe', pe)
        
    def forward(self,embed_vect):
        return embed_vect + self.pe


