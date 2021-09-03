# Import Numpy & PyTorch
import numpy as np
import torch

#Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')

# Target (apples)
targets = np.array([[56], 
                    [81], 
                    [119], 
                    [22], 
                    [103]], dtype='float32')


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

W = torch.rand((1,inputs.shape[1]),requires_grad=True)
b = torch.zeros(size=(inputs.shape[0],1),requires_grad=True)
print(W)

predict = lambda X,W,b : X@W.T + b

# Generate predictions
y_hat = predict(inputs,W, b)

# Compare with targets
print(y_hat)
print(targets)

# MSE loss
loss_fn = torch.nn.MSELoss()
custom_loss = lambda Y,Y_hat : ((Y-Y_hat)**2).mean()

# Compute loss
loss = loss_fn(targets,yhat)


# Compute gradients
loss.backward()

# Gradients for weights
print(W.grad)

W.grad.zero_(),b.grad.zero_()

X = torch.from_numpy( np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32'))
Y = torch.from_numpy( np.array([[56], 
                    [81], 
                    [119], 
                    [22], 
                    [103]], dtype='float32'))

W = torch.rand((1,X.shape[1]),requires_grad=True)
b = torch.tensor([[0.0]],requires_grad=True)

predict = lambda X,W,b : X@W.T + b
custom_loss = lambda Y,Yhat : ((Y-Yhat)**2).mean()

find_w_grad = lambda y_hat,Y,X:2*(y_hat-Y).T@X/len(Y)
find_b_grad = lambda y_hat,Y:sum(2*(y_hat-Y))/len(Y)

loss_fn = torch.nn.MSELoss()

W,b,W.grad,b.grad

# Generate predictions
predictions = predict(X,W,b)
predictions

# Calculate the loss
loss = loss_fn(Y,predictions)
loss

# Compute gradients
loss.backward()
print(W.grad,b.grad)

print(find_w_grad(predictions,Y,X),find_b_grad(predictions,Y))

alpha = 1e-3  

# Adjust weights & reset gradients
with torch.no_grad():
    W -= alpha*(W.grad)
    b -= alpha*(b.grad)
W.grad.zero_(),b.grad.zero_()
W,b

new_pred = predict(X,W,b)
# Calculate loss
loss = loss_fn(Y,new_pred)
loss

# Train for 100 epochs
epoch =150
alpha = 1e-5
W = torch.randn((1,X.shape[1]),requires_grad=True,dtype=torch.float32)
b = torch.randn((1,1),requires_grad=True,dtype=torch.float32)

for e in range(epoch):
    y_hat = predict(X,W,b)
    loss = loss_fn(Y,y_hat)
    loss.backward()
    with torch.no_grad():
        W -= alpha*W.grad
        b -= alpha*b.grad
    W.grad.zero_(),b.grad.zero_()

# Calculate loss
loss_fn(Y,yhat)

# Print targets
print(Y)