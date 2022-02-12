from predictor import Predictor
from training import learning_rate, batch_size, l1_neurons, l2_neurons, predict_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch 
import matplotlib.pyplot as plt
import random
import numpy as np

def testing():
    model = Predictor(l1_neurons, l2_neurons)
    model.load_state_dict(torch.load('model_params.pth'))
    test_dataset = MNIST(root='data', train=False, download=False,transform=transforms.ToTensor())

    img, label = test_dataset[random.randint(0, 100)]
    print(predict_image(img, model))
    plt.imshow(img[0])
    plt.show()

def second_test():
    model = Predictor(l1_neurons, l2_neurons)
    model.load_state_dict(torch.load('model_params.pth'))
 
    matrix = np.ones([28, 28], dtype='float32')
    print(matrix)




if __name__ == '__main__':
    # testing()
    second_test()