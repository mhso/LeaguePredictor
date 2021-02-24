import torch

i = torch.LongTensor([
    [0, 1, 2, 3, 4],
    [0, 0, 0, 0, 0]
])
v = torch.FloatTensor([3, 4, 5, 1, 2])
t = torch.sparse.FloatTensor(i, v, torch.Size([5, 371])).to_dense()

print(t)
