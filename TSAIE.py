import torch
from torch import nn
import torch.nn.functional as F
import consts
from VGGNet import VGGNet
from ResNet import ResNet
from transformers import AutoModel, AutoConfig, AutoTokenizer


class TSAIE(nn.Module):
    def __init__(self):
        super(TSAIE, self).__init__()
        self.bert = AutoModel.from_pretrained(consts.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(consts.MODEL_NAME)

        self.img_layer = nn.Sequential(
            ResNet(consts.hidden_size),
            nn.LayerNorm(consts.hidden_size),
        )

        self.dense = nn.Sequential(
            nn.ReLU(),
            nn.Linear(consts.hidden_size, 3),
        )
        # self.dense = nn.Sequential(
        #     nn.Linear(consts.hidden_size, 16),
        #     nn.Linear(16, 3),
        # )

        self.img_parameters = self.img_layer.parameters()
        self.txt_parameters = self.dense.parameters()

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X1, X2):  # X1 为图像, X2 为文字
        if self.is_cuda():
            X1, X2 = X1.cuda(), X2.cuda()

        B, S = X2.shape
        H = consts.hidden_size

        Ei = self.img_layer(X1)  # (B, H)

        mask = X2.ne(self.tokenizer.pad_token_id).bool()  # (B, S)
        key = self.bert.embeddings(X2)
        with torch.no_grad():
            output = self.bert(X2, attention_mask=mask).last_hidden_state
        if not mask.any():
            output = torch.full_like(output, 0)

        attn = F.cosine_similarity(Ei.view(B, 1, H), key, dim=2)
        attn = torch.softmax(attn, dim=1)
        Et = torch.sum(attn.view(B, S, 1) * output, dim=1)  # / seq_len.view(B, 1) 这里不能再除了

        # E = torch.cat([Et, Ei], dim=1)  # batch, hidden_size * 2

        res = self.dense(Et)
        return res
