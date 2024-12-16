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


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CasualAttention(d_in, d_out, context_lenght, dropout) for _ in range(num_heads)
            ])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)  # processing in for loop, each batch at once
    
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_lenght, dropout, num_heads, qkv_bias=False):
        super().__init__()  # super() function is used to call a method from a parent class in the inheritance hierarchy. 
                            # In the context of your MultiHeadAttention class, which inherits from torch.nn.Module, 
                            # the super().__init__() line is used to initialize the parent class (nn.Module) properly.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # reduce the projectsion dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # use Linear to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_lenght, context_lenght), diagonal=1)
        )   

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)  
         
                                    # Starting Tensor
                                    # x (shape: [1, 3, 8]) = 
                                    # [
                                    #     [ [1, 2, 3, 4, 5, 6, 7, 8],   # Token 1
                                    #       [9, 10, 11, 12, 13, 14, 15, 16],  # Token 2
                                    #       [17, 18, 19, 20, 21, 22, 23, 24]  # Token 3
                                    #     ]
                                    # ]
        

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      # splitting the matricies by adding the num_heads dimension. Then unrolling the last dim
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

                                                                            # Splitting into Heads
                                                                            # x (shape: [1, 3, 2, 4]) =
                                                                            # [
                                                                            #     [ 
                                                                            #       [ [1, 2, 3, 4],   # Head 1 for Token 1
                                                                            #         [5, 6, 7, 8]    # Head 2 for Token 1
                                                                            #       ],
                                                                            #       [ [9, 10, 11, 12],# Head 1 for Token 2
                                                                            #         [13, 14, 15, 16]# Head 2 for Token 2
                                                                            #       ],
                                                                            #       [ [17, 18, 19, 20],# Head 1 for Token 3
                                                                            #         [21, 22, 23, 24] # Head 2 for Token 3
                                                                            #       ]
                                                                            #     ]
                                                                            # ]

        keys = keys.transpose(1, 2)         # transpose from shape (b, num_tokens, num_heads, head_dim)
        queries = queries.transpose(1, 2)   # to shape (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  
                                            # Transposing ensures that each head processes its corresponding num_tokens and head_dim without interference.
                                            # This transposed shape allows efficient parallel computation for all heads.

                                            # Transposing for Attention
                                            # x (shape: [1, 2, 3, 4]) =
                                            # [
                                            #     [ 
                                            #       [ [1, 2, 3, 4],   # Token 1 for Head 1
                                            #         [9, 10, 11, 12],# Token 2 for Head 1
                                            #         [17, 18, 19, 20]# Token 3 for Head 1
                                            #       ],
                                            #       [ [5, 6, 7, 8],   # Token 1 for Head 2
                                            #         [13, 14, 15, 16],# Token 2 for Head 2
                                            #         [21, 22, 23, 24] # Token 3 for Head 2
                                            #       ]
                                            #     ]
                                            # ]

        attn_scores = queries @ keys.transpose(2, 3)    # dot products for each head. When we compute keys.transpose(2, 3), the shape:
                                                        # (batch_size, num_heads, num_tokens, head_dim) becomes:
                                                        # (batch_size, num_heads, head_dim, num_tokens)
                                                        # The matrix multiplication happens between the last two dimensions of queries and keys.transpose(2, 3)

                                                        # keys_transposed = [
                                                        #     [  # Head 1
                                                        #         [1, 5, 9],  # Token 1
                                                        #         [2, 6, 10], # Token 2
                                                        #         [3, 7, 11], # Token 3
                                                        #         [4, 8, 12]  # Token 4
                                                        #     ],
                                                        #     [  # Head 2
                                                        #         [13, 17, 21],  # Token 1
                                                        #         [14, 18, 22],  # Token 2
                                                        #         [15, 19, 23],  # Token 3
                                                        #         [16, 20, 24]   # Token 4
                                                        #     ]
                                                        # ]

                                                        # attn_scores = [
                                                        #     [  # Head 1
                                                        #         [90, 100, 110],  # Token 1
                                                        #         [200, 220, 240], # Token 2
                                                        #         [310, 340, 370]  # Token 3
                                                        #     ],
                                                        #     [  # Head 2
                                                        #         [520, 560, 600],  # Token 1
                                                        #         [640, 680, 720],  # Token 2
                                                        #         [760, 800, 840]   # Token 3
                                                        #     ]
                                                        # ]

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
                                                        # attn_scores = [
                                                        #     [  # Head 1
                                                        #         [90, -inf, -inf],  # Token 1 (masked out Token 2 and Token 3)
                                                        #         [200, 220, -inf],  # Token 2 (masked out Token 3)
                                                        #         [310, 340, 370]    # Token 3 (no mask for Token 3)
                                                        #     ],
                                                        #     [  # Head 2
                                                        #         [520, -inf, -inf],  # Token 1 (masked out Token 2 and Token 3)
                                                        #         [640, 680, -inf],   # Token 2 (masked out Token 3)
                                                        #         [760, 800, 840]     # Token 3 (no mask for Token 3)
                                                        #     ]
                                                        # ]

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**2, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   # tensor shape -> (b, _num_tokens, n_heads, head_dim)

                                                                # attn_weights
                                                                # Shape: (batch_size, num_heads, num_tokens, num_tokens)
                                                                # Rows: Attention weights for each query token.
                                                                # Columns: How much attention each query pays to every key token.

                                                                # values
                                                                # Shape: (batch_size, num_heads, num_tokens, head_dim)
                                                                # For each token, the "value" vectors hold information to pass forward.

                                                                # The operation attn_weights @ values computes the weighted sum of the values for each query token.
                                                                # Shape after multiplication:
                                                                # (batch_size, num_heads, num_tokens, head_dim)
                                                                # For each query token, we aggregate the values weighted by the attention scores.

                                                                #  context_vec = [
                                                                #     [  # Token 1
                                                                #         [1, 2, 3, 4],      # Head 1
                                                                #         [13, 14, 15, 16]   # Head 2
                                                                #     ],
                                                                #     [  # Token 2
                                                                #         [5.53, 6.53, 7.53, 8.53],  # Head 1
                                                                #         [18.47, 19.47, 20.47, 21.47]  # Head 2
                                                                #     ],
                                                                #     [  # Token 3
                                                                #         [9, 10, 11, 12],   # Head 1
                                                                #         [21, 22, 23, 24]   # Head 2
                                                                #     ]
                                                                # ]

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # combine heads where self.d_out = self.num_heads * self.head_dim

                                                                # context_vec = [
                                                                #     [  # Batch 1
                                                                #         [1, 2, 3, 4, 13, 14, 15, 16],  # Token 1
                                                                #         [5.53, 6.53, 7.53, 8.53, 18.47, 19.47, 20.47, 21.47],  # Token 2
                                                                #         [9, 10, 11, 12, 21, 22, 23, 24]  # Token 3
                                                                #     ]
                                                                # ]

        context_vec = self.out_proj(context_vec)    # Linear projection

        return context_vec
        




