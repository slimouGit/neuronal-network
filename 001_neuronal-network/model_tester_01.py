import torch
from nn_builder import NeuralNetwork

# Create the model (50 inputs, 3 outputs)
model = NeuralNetwork(50, 3)

# Generate a random input sample
sample = torch.randn(1, 50)

# Run inference
with torch.no_grad():
    logits = model(sample)
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()

print("Predicted probabilities:", probs.numpy())
print("Predicted class:", predicted_class)