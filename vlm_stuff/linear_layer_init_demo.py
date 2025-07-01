import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleProjector(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, output_size, bias=True)
        # self.act = F.glu
        self.linear_2 = nn.Linear(output_size, output_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        # hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# Define the size of the input and output for the linear layer
input_size = 1
output_size = 2

# Create a linear layer with Xavier initialization
linear_layer = nn.Linear(input_size, output_size, bias=True)
projector = ExampleProjector()

# input_tensors = torch.randn(input_size)
input_tensors = torch.tensor([5], dtype=torch.float)
# input_tensors = torch.full((input_size,), 5)
print(input_tensors)
result = projector(input_tensors)
print(projector.linear_1.weight)
print(result)

quit()

# Print the weights of the linear layer
print("Weights before Xavier initialization:")
print(linear_layer.weight)

# Apply Xavier initialization
nn.init.xavier_uniform_(linear_layer.weight)

# Print the weights after Xavier initialization
print("\nWeights after Xavier initialization:")
print(linear_layer.weight)
