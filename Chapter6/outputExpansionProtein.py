import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

data_path = '/Users/beyzakaya/Desktop/bk/Proje:Kod/Chapter3/Chapter6/datasets/training_secondary_structure_train.csv'
df = pd.read_csv(data_path)
#print(len(df))
sequences = df['seq'].tolist()
#print(f"Sequence example: {len(sequences)}")
labels = df['sst3'].tolist()
#print(f"Label examples: {len(labels)}")

# for i in range(5):
#     print(f"Sequence {i}: {sequences[i]}")
#     print(f"Label {i}: {labels[i]}")
#     print(f"Sequence length: {len(sequences[i])}, Label length: {len(labels[i])}")
#     print()

def one_hot_encode_sequences(sequences):
    chars = sorted(set(''.join(sequences)))
    char_to_index = {char: i for i, char in enumerate(chars)}

    encoder = OneHotEncoder(sparse_output=False, categories=[chars])

    encoded_sequences = []
    max_length = max(len(seq) for seq in sequences)  
    for seq in sequences:
        encoded_seq = encoder.fit_transform(np.array(list(seq)).reshape(-1, 1))
    
        if len(seq) < max_length:
            padding = np.zeros((max_length - len(seq), encoded_seq.shape[1]))
            encoded_seq = np.vstack([encoded_seq, padding])
        
        encoded_sequences.append(encoded_seq.flatten())
    
    return np.array(encoded_sequences), char_to_index

encoded_sequences, char_to_index = one_hot_encode_sequences(sequences)

def encode_labels(labels):
    label_map = {'H': 0, 'E': 1, 'C': 2}
    numerical_labels = [np.array([label_map[char] for char in label]) for label in labels]
    return numerical_labels

numerical_labels = encode_labels(labels)

#print(f"Numerical labels: {numerical_labels}")

train_sequences, test_sequences, train_labels, test_labels = train_test_split(encoded_sequences,numerical_labels,test_size=0.2, random_state=42)

def prepare_data(sequences, labels, window_size=15, output_window_size=5):
    X, y = [], []
    half_window = window_size // 2
    
    for seq, label in zip(sequences, labels):
        seq_len = len(seq) // len(set(seq))  # Adjusted for correct length calculation
        label_len = len(label)
        
        if seq_len != label_len:
            continue
        
        for i in range(half_window, seq_len - half_window):
            input_window = seq[i - half_window:i + half_window + 1]
            start_idx = i - output_window_size // 2
            end_idx = i + output_window_size // 2 + 1
            
            if start_idx < 0 or end_idx > label_len:
                continue
            
            output_window = label[start_idx:end_idx]
            
            if len(output_window) == output_window_size:
                X.append(input_window)
                y.append(output_window)
    X = np.array(X)
    y = np.array(y)
    print(f"Prepared X shape: {X.shape}")
    print(f"Prepared y shape: {y.shape}")
    
    return X, y


window_size = 15
output_window_size = 5
train_X, train_y = prepare_data(train_sequences, train_labels, window_size, output_window_size)
test_X, test_y = prepare_data(test_sequences, test_labels, window_size, output_window_size)
print(f"train_X shape: {train_X.shape}")
print(f"train_y shape: {train_y.shape}")

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.long)

class OutputExpansionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_window_size):
        super(OutputExpansionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim * output_window_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
input_dim = train_X.shape[1]
hidden_dim = 128
output_dim = 3
model = OutputExpansionModel(input_dim, hidden_dim, output_dim, output_window_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_X)
    outputs = outputs.view(-1, output_window_size, output_dim)
    train_y = train_y.view(-1, output_window_size)
    loss = criterion(outputs.view(-1, output_dim), train_y.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(test_X)
    test_outputs = test_outputs.view(-1, output_window_size, output_dim)
    test_y = test_y.view(-1, output_window_size)
    test_loss = criterion(test_outputs.view(-1, output_dim), test_y.view(-1))
    print(f"Test Loss: {test_loss.item():.4f}")
    
    _, predicted = torch.max(test_outputs, 2)
    predicted = predicted.view(-1).numpy()
    test_y_np = test_y.view(-1).numpy()
    
    correct = (predicted == test_y_np).sum()
    total = test_y_np.size
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")