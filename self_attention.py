import torch.nn as nn
import torch

class SelfAttention_V1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values

        return context_vector
    
class SelfAttention_V2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # uses more sophisticated weight initialization scheme than random nn.Parameter(torch.rand()) method
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)    # also, it stores matricies in transposed form
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)


    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
            )
        context_vector = attn_weights @ values

        return context_vector


class SelfAttention_V1_with_V2_weights(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        sa_v2 = SelfAttention_V2(d_in, d_out)

        self.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
        self.W_key   = torch.nn.Parameter(sa_v2.W_key.weight.T)
        self.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
        
    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values

        return context_vector
    

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, contex_lenght, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.d_in  = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(contex_lenght, contex_lenght), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values

        return context_vector
