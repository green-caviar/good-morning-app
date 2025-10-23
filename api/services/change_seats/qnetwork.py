import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        """ネットワークの層（部品）をここで定義する"""

        super(QNetwork, self).__init__()

        # 3層のシンプルなネットワークを定義
        self.layer1 = nn.Linear(state_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_size)

    def forward(self, state):
        """データがネットワークを流れる「順序」をここで定義する"""

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        return self.layer3(x)