import torch
from nn_builder import NeuralNetwork

# Create the model (50 inputs, 3 outputs)
model = NeuralNetwork(50, 3)

# Example: three synthetic samples for three classes
samples = torch.zeros(3, 50)
samples[0, 0:10] = 1.0  # Sample 1: features 0-9 active
samples[1, 10:20] = 1.0 # Sample 2: features 10-19 active
samples[2, 20:30] = 1.0 # Sample 3: features 20-29 active

# Run inference
with torch.no_grad():
    logits = model(samples)
    probs = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probs, dim=1)

for i, pred in enumerate(predicted_classes):
    print(f"Sample {i+1}: Predicted class {pred.item()}, Probabilities: {probs[i].numpy()}")