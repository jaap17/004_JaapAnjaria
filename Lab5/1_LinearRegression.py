#import numpy and torch
import numpy as np
import torch

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')

# Target (apples)
targets = np.array([[56], [81], [119], [22], [103]], dtype='float32')

# Convert inputs and targets to tensors
x_inputs = torch.from_numpy(inputs)
print(x_inputs)
x_targets = torch.from_numpy(targets)
print(x_targets)

# Weights and biases

