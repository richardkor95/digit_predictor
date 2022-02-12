import torch 
import torchvision 
from torchvision.datasets import MNIST 
import torchvision.transforms as transforms
from predictor import Predictor, accuracy, evaluate
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

saving = True

# hyperparameter
optimizer = torch.optim.SGD
learning_rate = 0.005
l1_neurons = 28*28
l2_neurons = 28*28
training_epochs = 20
batch_size = 128

def training(epochs, lr, model, batch_size, opt_func=torch.optim.SGD):
    print(model.parameters())
    optimizer = opt_func(model.parameters(), lr)

    # downloading the data of not already existing 
    if not Path('data') .is_dir():
        dataset = MNIST(root='data/', download=True)
    else:
        dataset = MNIST(root='data/', download=False)         

    # define test dataset 
    test_dataset = MNIST(root='data', train=False)
    
    # Transform to Tensor
    dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())

    # split dataset in training and validation set 
    train_ds, val_ds = random_split(dataset, [50000, 10000])

    # create the data batches 
    batch_size = 128
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size)
        
    # trainingloop
    for epoch in range(epochs): 
        if epoch > 10: 
            lr = 0.001
            optimizer = opt_func(model.parameters(), lr)
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)

    return model


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

def pick_random_datapoint():
    testset = MNIST(root='data/', 
            train=False,
            transform=transforms.ToTensor())
    
    img, label = testset[0]
    plt.imshow(img[0])
    plt.show()

def make_prediction(matrix):
    model = Predictor(l1_neurons, l2_neurons)
    model.load_state_dict(torch.load('model_params.pth'))
    img = torch.from_numpy(matrix)
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    model.make_prediction(img)


if __name__== '__main__':
    model = Predictor(l1_neurons, l2_neurons)
    make_new_model = True
    if not make_new_model:
        model.load_state_dict(torch.load('model_params.pth'))

    model = training(training_epochs, learning_rate, model, batch_size)
    torch.save(model.state_dict(), 'model_params.pth')
    
