import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 28*28)
        self.l2 = nn.Linear(28*28, 10)
        self.actfun = F.relu

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.l1(x)
        x = self.actfun(x)
        x = self.l2(x)
        return x

    def make_prediction(self, image):
        image = image.reshape(-1, 28*28)
        out = self(image)
        out = F.softmax(out)
        cert, number = torch.max(out, dim=1)
        return {'Output': int(number.numpy()), 'certainty': cert}

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    

