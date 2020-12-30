from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from torch import Tensor,save
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
from model import Model
from utils import *
#from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    X_train = [rgb2gray(image) for image in X_train]
    X_train = np.array(X_train)
    X_valid = [rgb2gray(image) for image in X_valid]
    X_valid = np.array(X_valid)
    
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
    
    # Action to id
    y_train = [action_to_id(action) for action in y_train]
    y_train = np.array(y_train)
    y_valid = [action_to_id(action) for action in y_valid]
    y_valid = np.array(y_valid)
    
    # One-hot encoding
    #y_train = one_hot(y_train)
    #y_valid = one_hot(y_valid)

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, momentum, epochs, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")

    # TODO: specify your neural network in model.py 
    agent = Model()
    # TODO: implement the training    
    optimizer = optim.SGD(agent.parameters(), lr=lr, momentum=momentum)
    training_losses = []
    
    train_data = util_data.TensorDataset(Tensor(X_train), Tensor(y_train))
    val_data = util_data.TensorDataset(Tensor(X_valid), Tensor(y_valid))
    train_data = util_data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    val_data = util_data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)
    for epoch in range(1, epochs + 1):
        train(agent, train_data, optimizer, epoch, training_losses)
        validation(agent, val_data)
    # TODO: save your agent
    model_path = os.path.join(model_dir, "agent.ckpt")
    save(agent.state_dict(), os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_path)
    plt.plot(training_losses)

def train(model, train_data, optimizer, epoch, training_losses):
    model.train()
    for batch, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(data.unsqueeze(1))
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(data), len(train_data.dataset),
                100. * batch / len(train_data), loss.item()))
    training_losses.append(loss.item())

def validation(model, val_data):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_data:
        output = model(data.unsqueeze(1))
        validation_loss += F.nll_loss(output, target.long(), reduction="sum").item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_data.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_data.dataset),
        100. * correct / len(val_data.dataset)))

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=0.0001, momentum=0.9, epochs=100)
 
