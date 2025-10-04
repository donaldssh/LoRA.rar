from torch import nn


class Hypernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_1 = nn.Linear(640*2, 128)         
        self.input_2 = nn.Linear(1280*2, 128)         
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        if x.size()[1] == 640*2:
            x = self.input_1(x)
        elif x.size()[1] == 1280*2:
            x = self.input_2(x)
        return self.output(x)

