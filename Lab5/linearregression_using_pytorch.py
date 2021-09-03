# Import Numpy & PyTorch
import numpy as np
import torch

# Imports
import torch.nn as nn

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')


X = torch.from_numpy(inputs)
Y = torch.from_numpy(targets)

# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader

# Define dataset
dataset = torch.utils.data.TensorDataset(X,Y)

# Define data loader
trainloader = torch.utils.data.DataLoader(dataset,batch_size=10)


# Define model
model = nn.Linear(X.shape[1],Y.shape[1])
print(model.state_dict())
model.parameters()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)

#Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.
# Import nn.functional
import torch.nn.functional as F

# Define loss function
loss_fn = F.mse_loss

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in trainloader:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(yb,pred)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(X), Y))


# Train the model for 50 epochs
fit(50 , model , loss_fn, optimizer)


# Generate predictions
preds = model(X)
preds

# Compare with targets
targets




#Exercise 1: Try Linear Regression just using numpy (Without Tensorflow/Pytorch or other torch library)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()

from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test = train_test_split(inputs,targets,test_size=0.2,random_state=142)

model1.fit(X_Train,Y_Train)

Y_pred = model1.predict(X_Test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_Test,Y_pred))
print(Y_pred,Y_Test)