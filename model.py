import torch
import torch.nn as nn
import math as m

class InputEmbeddings(nn.Module): # class inheritence for new instance of nn per layer of model
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__() # initializes inherited class
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # number of tokens and vector size for embedding matrix shape of (vocab_size by d_model)

    def forward(self, x): # matrix shape of (x by d_model)
        return self.embedding(x) * m.sqrt(self.d_model) # x is nothing but the number of tokens


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # dropout function uses a trained value to deactivate that number of nuerons to avoid overfitting

        # Creating the matrix of shape (seq_length x d_model)
        pe = 
