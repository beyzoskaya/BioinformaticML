import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

codon_to_amino_acid = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': 'Stop', 'UAG': 'Stop',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'UGU': 'C', 'UGC': 'C', 'UGA': 'Stop', 'UGG': 'W',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

def one_hot_encode_codon(codon):
    base_map = {'A': [0, 0, 0, 1], 'C': [0, 0, 1, 0], 'G': [0, 1, 0, 0], 'U': [1, 0, 0, 0]}
    return np.array(base_map[codon[0]] + base_map[codon[1]] + base_map[codon[2]])

def amino_acid_to_one_hot(amino_acid):
    amino_acids = 'FLSYCWPHQRIMTNKVADEG'
    one_hot = np.zeros(len(amino_acids))
    index = amino_acids.index(amino_acid)
    one_hot[index] = 1
    return one_hot

inputs = []
outputs = []

for codon,amino_acid in codon_to_amino_acid.items():
    if amino_acid != 'Stop':
        inputs.append(one_hot_encode_codon(codon))
        outputs.append(amino_acid_to_one_hot(amino_acid))

inputs = np.array(inputs)
print(f"Inputs: {inputs}")
print(f"Len of inputs: {len(inputs)}")
outputs = np.array(outputs)
print(f"Outputs: {outputs}")
print(f"Len of outputs: {len(outputs)}")

inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

class GeneticCodeNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneticCodeNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
input_dim = 12  # 12-bit encoding for codons
hidden_dim = 2  # Start with 2 hidden units 
output_dim = 20  # 20 possible amino acids

model = GeneticCodeNN(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    
    outputs_pred = model(inputs)
    
    loss = criterion(outputs_pred, outputs)
    loss.backward()
    
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(inputs)
    predicted_amino_acids = torch.argmax(predictions, dim=1)
    actual_amino_acids = torch.argmax(outputs, dim=1)

    accuracy = (predicted_amino_acids == actual_amino_acids).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")