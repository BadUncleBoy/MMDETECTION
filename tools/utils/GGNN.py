import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state, A):
        
        a0 = torch.mm(A, state)

        a = torch.cat((a0, state), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a0, r * state), 1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state + z * h_hat

        return output

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, A, fc_out_channels, feature_dim=20, state_dim=10, n_steps=5):
        super(GGNN, self).__init__()
        self.n_steps = n_steps

        self.A = A
        self.classifier_weight = Variable(torch.rand(feature_dim, fc_out_channels).cuda(), requires_grad=True)
        #convert input node dimension to state_dim
        self.feature2state_dim = nn.Linear(fc_out_channels, state_dim)
        self.state2feature_dim = nn.Linear(state_dim, fc_out_channels)
        # Propogation Model
        self.propogator = Propogator(state_dim)
        self._initialization()

    def _initialization(self):
        self.classifier_weight.data.normal_(0.0, 0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, feat):
        state = self.feature2state_dim(self.classifier_weight)

        for _ in range(self.n_steps):
            state = self.propogator(state, self.A)
        
        result = self.state2feature_dim(state)
        return torch.mm(feat, result.permute(1, 0))
 
if __name__ == "__main__":
    Adjecent = torch.ones(1001,1001).cuda()
    feat = torch.rand(512,1024).cuda()
    ggnn =GGNN(Adjecent,1024,1001,512,2)
    ggnn = ggnn.cuda()
    res = ggnn(feat)
    loss = torch.sum(res)
    loss.backward()
    print(ggnn.classifier_weight.grad)
    print(res.size())
