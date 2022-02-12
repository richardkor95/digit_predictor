import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):

    l1_neurons = 28*28
    l2_neurons = 28*28

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, self.l1_neurons)
        self.l2 = nn.Linear(self.l2_neurons, 10)
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
        out = F.softmax(out[0], dim=0)
        cert, number = torch.max(out, dim=0)
        print(f'Prediction: {int(number.numpy())}, Certainty: {round(cert.detach().numpy()*100,2)}, 2 %')

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                   
        loss = F.cross_entropy(out, labels)  
        acc = accuracy(out, labels)          
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def make_prediction(matrix):
    model = Predictor()
    model.load_state_dict(torch.load('model_params.pth'))
    img = torch.from_numpy(matrix)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    model.make_prediction(img)