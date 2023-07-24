# Author: Zeyu Wang
# Department: Sun Yat-sen University
# Function: (LCGNN) Generate LC signal

import torch.nn as nn

class LCGNN(nn.Module):
    def __init__(self, inputChannel=3, outputChannel=2):
        super(LCGNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputChannel, 64, kernel_size=9, padding=4, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, outputChannel, kernel_size=5, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

model = LCGNN()
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(model))
