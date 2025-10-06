import torch
from nn_builder import NeuralNetwork

# Create the model (50 inputs, 3 outputs)
model = NeuralNetwork(50, 3)

# Synthetic samples for each class
samples = torch.zeros(3, 50)
labels = torch.tensor([0, 1, 2])  # True classes

samples[0, 0:10] = 1.0   # Class 0 pattern
samples[1, 10:20] = 1.0  # Class 1 pattern
samples[2, 20:30] = 1.0  # Class 2 pattern

# Run inference
with torch.no_grad():
    logits = model(samples)
    probs = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)

for i, (pred, true) in enumerate(zip(predicted_classes, labels)):
    print(f"Sample {i+1}: True class {true}, Predicted class {pred.item()}, Probabilities: {probs[i].numpy()}")