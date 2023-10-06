import torch.nn as nn
import torch
class Te(nn.Module):
    def _init_(self):
        super().__init__()
    def forward(self,x):
        output = x+1
        return output
te=Te()
x=torch.tensor(3)
output=te(x)
print(output)