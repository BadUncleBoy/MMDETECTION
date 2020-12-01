#self add
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from .convfc_bbox_head import SharedFCBBoxHead
from mmdet.models.builder import HEADS

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
    def __init__(self, A, fc_out_channels, init_weights, state_dim=10, n_steps=5):
        super(GGNN, self).__init__()
        self.n_steps = n_steps

        self.A = A
        self.classifier_weight = init_weights
        #convert input node dimension to state_dim
        self.feature2state_dim = nn.Sequential(
                                    nn.Linear(init_weights.size()[-1], state_dim),
                                    nn.Tanh())
        self.state2feature_dim = nn.Sequential(
                                    nn.Linear(state_dim, fc_out_channels))
        # Propogation Model
        self.propogator = Propogator(state_dim)
        self._initialization()

    def _initialization(self):
        # nn.init.normal_(self.classifier_weight, 0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        state = self.feature2state_dim(self.classifier_weight)

        for _ in range(self.n_steps):
            state = self.propogator(state, self.A)
        
        result = self.state2feature_dim(state)
        return torch.mm(feat, result.permute(1, 0))


@HEADS.register_module
class GGNNBBOXHEAD(SharedFCBBoxHead):
    def __init__(self,
                 num_fcs=2,
                 ggnn_config=None,
                 fc_out_channels=1024,
                 *args,
                 **kwargs):
        super(GGNNBBOXHEAD, self).__init__(num_fcs=num_fcs,
                                         fc_out_channels=fc_out_channels,
                                         *args,
                                         **kwargs)
        adjecent = torch.load(ggnn_config.adjecent_path).cuda()
        init_weights = torch.load(ggnn_config.initweight_path).cuda()
        self.fc_cls = GGNN(A=adjecent, 
                           fc_out_channels=fc_out_channels,
                           init_weights=init_weights, 
                           state_dim=ggnn_config.state_dim,
                           n_steps=ggnn_config.n_steps)
        print("called")
        print(self.modules)
    def init_weights(self):
        # 重写权重初始化函数，在GGNN中已经对分类网络进行权重初始化了
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
