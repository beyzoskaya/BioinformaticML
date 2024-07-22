import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim

data_path = '/Users/beyzakaya/Desktop/bk/Proje:Kod/Chapter3/Chapter6/training_secondary_structure_train.csv'
df = pd.read_csv(data_path)
sequences = df['seq'].tolist()
print(f"Length of sequences: {len(sequences)}")
sequences = [seq.replace('X', 'A') for seq in sequences]

def generate_synthetic_labels(sequences, num_classes=3):
    labels = []
    for seq in sequences:
        label = np.random.randint(0, num_classes, len(seq))
        labels.append(label)
    return labels

synthetic_labels = generate_synthetic_labels(sequences, num_classes=3)

def encode_sequences(sequences):
    encoder = OneHotEncoder(categories=[list('ACDEFGHIKLMNPQRSTVWY')], sparse_output=False)
    encoded_sequences = [encoder.fit_transform(np.array(list(seq)).reshape(-1, 1)).mean(axis=0) for seq in sequences]
    return encoded_sequences

pssm_encoded_sequences = encode_sequences(sequences)

class PSIPRED_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PSIPRED_NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(pssm, labels, epochs=10):
    num_residues = pssm[0].shape[0]
    model = PSIPRED_NN(num_residues, 50, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for pssm_seq, label_seq in zip(pssm, labels):
            pssm_tensor = torch.tensor(pssm_seq, dtype=torch.float32).unsqueeze(0) 
            labels_tensor = torch.tensor(label_seq, dtype=torch.long).unsqueeze(0) 

            optimizer.zero_grad()
            outputs = model(pssm_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(pssm)}')

train_model(pssm_encoded_sequences, synthetic_labels)
