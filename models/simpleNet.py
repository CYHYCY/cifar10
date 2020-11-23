import torch.nn as nn
import torch


class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class YHT_Net(nn.Module):
    def __init__(self, in_dim=3 * 32 * 32, n_hidden_1=1024, n_hidden_2=512, n_hidden_3=128, out_dim=10, use_bias=True):
        super(YHT_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1, bias=use_bias), nn.BatchNorm1d(n_hidden_1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2, bias=use_bias), nn.BatchNorm1d(n_hidden_2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3, bias=use_bias), nn.BatchNorm1d(n_hidden_3),
                                    nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim, bias=use_bias))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == "__main__":
    model = YHT_Net(3 * 32 * 32, 512, 256, 128, 10)
    x = torch.randn(size=(2, 3, 32, 32))
    x = x.view(x.size(0), -1)
    y = model(x)
    print(y.size())
