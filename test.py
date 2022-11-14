import torch
m = torch.nn.Linear(-1, 30)
input = torch.randn(128, 20)#输入数据的维度(128,20)
output = m(input)
print(m.weight.shape)
print(m.bias.shape)
print(output.size())