import torch

from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.o_linear = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X)
        bs, s, hidden_dim = X.shape
        q, k, v = self._split_heads(q, self.num_heads), self._split_heads(k, self.num_heads), self._split_heads(v, self.num_heads)
        mask = torch.ones((1, 1, s, s), dtype=torch.bool, device=X.device)
        mask = torch.tril(mask)
        attn = self._attention(q, k, v, mask)
        # attn: (bs, num_heads, s, head_dim)
        out = attn.transpose(1, 2).reshape(bs, s, hidden_dim)
        out = self.o_linear(out)
        return out
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q, k: (bs, n_heads, s, head_dim)
        # scores: (bs, n_heads, s, s)
        scaled = (self.hidden_dim // self.num_heads) ** 0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) / scaled
        if mask is not None:
            scores.masked_fill_(mask == False, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        # out: (bs, n_heads, s, head_dim)
        return torch.matmul(scores, v)
    def _split_heads(self, X: torch.Tensor, num_heads) -> torch.Tensor:
        # X: (bs, s, d) -> out: (bs, num_heads, s, head_dim)
        bs, s, hidden_dim = X.shape
        head_dim = hidden_dim // num_heads
        X = X.reshape(bs, s, num_heads, head_dim).transpose(1, 2)
        return X

class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        super(MultiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, self.head_dim)
        self.v_linear = nn.Linear(hidden_dim, self.head_dim)
        self.o_linear = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X)
        bs, s, d = X.shape
        q = self._split_heads(q, self.num_heads)
        k = k.view(bs, 1, s, self.head_dim).expand(-1, self.num_heads, -1, -1)
        v = v.view(bs, 1, s, self.head_dim).expand(-1, self.num_heads, -1, -1)
        mask = torch.ones((1, 1, s, s), dtype=torch.bool, device=X.device)
        mask = torch.tril(mask)
        attn = self._attention(q, k, v, mask)
        out = attn.transpose(1, 2).reshape(bs, s, d)
        out = self.o_linear(out)
        return out
    
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q, k: (bs, n_heads, s, head_dim)
        # scores: (bs, n_heads, s, s)
        scaled = (self.hidden_dim // self.num_heads) ** 0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) / scaled
        if mask is not None:
            scores.masked_fill_(mask == False, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        # out: (bs, n_heads, s, head_dim)
        return torch.matmul(scores, v)
    def _split_heads(self, X: torch.Tensor, num_heads) -> torch.Tensor:
        # X: (bs, s, d) -> out: (bs, num_heads, s, head_dim)
        bs, s, hidden_dim = X.shape
        head_dim = hidden_dim // num_heads
        X = X.reshape(bs, s, num_heads, head_dim).transpose(1, 2)
        return X

class GroupQueryAttention(nn.Module):
    def __init__(self, num_heads: int, num_groups: int, hidden_dim: int):
        super(GroupQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0
        assert num_heads % num_groups == 0
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.v_linear = nn.Linear(hidden_dim, num_groups * self.head_dim)
        self.o_linear = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bs, s, d = X.shape
        q, k, v = self.q_linear(X), self.k_linear(X), self.v_linear(X)
        q = self._split_heads(q, self.num_heads)
        k = k.reshape(bs, s, self.num_groups, self.head_dim).unsqueeze(2)
        k = k.expand(-1, -1, self.num_heads // self.num_groups, -1, -1)
        k = k.reshape(bs, self.num_heads, s, self.head_dim)
        v = v.reshape(bs, s, self.num_groups, self.head_dim).unsqueeze(2)
        v = v.expand(-1, -1, self.num_heads // self.num_groups, -1, -1)
        v = v.reshape(bs, self.num_heads, s, self.head_dim)
        mask = torch.ones((1, 1, s, s), dtype=torch.bool, device=X.device)
        mask = torch.tril(mask)
        attn = self._attention(q, k, v, mask)
        out = attn.transpose(1, 2).reshape(bs, s, d)
        out = self.o_linear(out)
        return out
        
    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q, k: (bs, n_heads, s, head_dim)
        # scores: (bs, n_heads, s, s)
        scaled = (self.hidden_dim // self.num_heads) ** 0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) / scaled
        if mask is not None:
            scores.masked_fill_(mask == False, float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        # out: (bs, n_heads, s, head_dim)
        return torch.matmul(scores, v)
    def _split_heads(self, X: torch.Tensor, num_heads) -> torch.Tensor:
        # X: (bs, s, d) -> out: (bs, num_heads, s, head_dim)
        bs, s, hidden_dim = X.shape
        head_dim = hidden_dim // num_heads
        X = X.reshape(bs, s, num_heads, head_dim).transpose(1, 2)
        return X

def main():
    bs, s, d = 32, 128, 768
    X = torch.randn(bs, s, d)
    mha = MultiHeadAttention(4, d)
    out = mha(X)
    print('MultiHeadAttention:')
    print(out.shape)
    
    mqa = MultiQueryAttention(4, d)
    out = mqa(X)
    print('MultiQueryAttention:')
    print(out.shape)
    
    gqa = GroupQueryAttention(4, 2, d)
    out = gqa(X)
    print('Group Query Attention:')
    print(out.shape)
    

if __name__ == "__main__":
    main()