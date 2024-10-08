import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
alphabet_size = len(amino_acids) + 1

def generate_sample_sequences(num_sequences, sequence_length):
    sequences = []
    for _ in range(num_sequences):
        sequence = np.random.choice(amino_acids, sequence_length).tolist()
        sequences.append(sequence)
    return sequences

def create_orthogonal_encoding(amino_acids, alphabet_size):
    encoding_map = {}
    for i, aa in enumerate(amino_acids):
        encoding = np.zeros(alphabet_size)
        encoding[i] = 1
        encoding_map[aa] = encoding
    encoding_map['<TERMINATOR>'] = np.zeros(alphabet_size)
    encoding_map['<TERMINATOR>'][-1] = 1
    return encoding_map

encoding_map = create_orthogonal_encoding(amino_acids, alphabet_size)

def encode_sequence(sequence, encoding_map, window_size):
    padded_sequence = ['<TERMINATOR>'] * ((window_size - 1) // 2) + sequence + ['<TERMINATOR>'] * ((window_size - 1) // 2)
    encoded_sequence = []
    for i in range(len(padded_sequence) - window_size + 1):
        window = padded_sequence[i:i + window_size]
        encoded_window = [encoding_map[residue] for residue in window]
        encoded_sequence.append(np.concatenate(encoded_window, axis=0))
    return np.array(encoded_sequence)

alpha_helix_aa = {'Ala', 'Leu', 'Met', 'Glu', 'Lys'}
beta_sheet_aa = {'Val', 'Ile', 'Phe', 'Tyr', 'Cys'}
coil_aa = {'Gly', 'Pro', 'Ser', 'Asn', 'Gln'}

def label_sequence(sequence):
    alpha_count = sum(aa in alpha_helix_aa for aa in sequence)
    beta_count = sum(aa in beta_sheet_aa for aa in sequence)
    coil_count = sum(aa in coil_aa for aa in sequence)

    if alpha_count > beta_count and alpha_count > coil_count:
        return 0  # alpha-helix structure
    elif beta_count > alpha_count and beta_count > coil_count:
        return 1  # beta-sheet
    else:
        return 2  # coil

num_sequences = 500
sequence_length = 30
window_size = 13

sequences = generate_sample_sequences(num_sequences, sequence_length)

encoded_sequences = []
labels = []
for seq in sequences:
    encoded_seq = encode_sequence(seq, encoding_map, window_size)
    encoded_sequences.append(encoded_seq)
    label = label_sequence(seq)
    labels.extend([label] * encoded_seq.shape[0])

encoded_sequences = np.concatenate(encoded_sequences, axis=0)
labels = np.array(labels)

label_counts = Counter(labels)
print(f"Number of label 0s (alpha-helix): {label_counts[0]}")
print(f"Number of label 1s (beta-sheet): {label_counts[1]}")
print(f"Number of label 2s (coil): {label_counts[2]}")

X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class AdvancedMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(AdvancedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x
    
def prepare_data_loader(X, y, batch_size):
    dataset = TensorDataset(torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 16
train_loader = prepare_data_loader(X_train, y_train, batch_size)
test_loader = prepare_data_loader(X_test, y_test, batch_size)

input_size = window_size * alphabet_size
hidden_size = 40
output_size = 3

hidden_size1 = 128
hidden_size2 = 64
model = AdvancedMLP(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, output_size=output_size)

#model = SimpleMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

losses = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_plot_AdvncedMLP_Secondary_struc.png')
plt.show()

model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

report = classification_report(y_true, y_pred, target_names=['alpha-helix', 'beta-sheet', 'coil'])
print("Classification Report:")
print(report)

cm = confusion_matrix(y_true,y_pred,labels=[0,1,2])
PX = np.diag(cm) # True Positive
PfX = np.sum(cm, axis=0) - PX # False Positive
NX = np.sum(cm, axis=1) - PX
NfX = np.sum(cm) - (PX + PfX + NX)
C_X = np.zeros(3)  
for i in range(3):
    numerator = (PX[i] * NX[i]) - (NfX[i] * PfX[i])
    denominator = ((NX[i] + NfX[i]) * (NX[i] + PfX[i]) * (PX[i] + NfX[i]) * (PX[i] + PfX[i]))
    if denominator == 0:
        C_X[i] = 0
    else:
        C_X[i] = numerator / denominator

print("Confusion Matrix:\n", cm)
print(f"CX for alpha-helix: {C_X[0]}")
print(f"CX for beta-sheet: {C_X[1]}")
print(f"CX for coil: {C_X[2]}")

"""
Confussion Matrix for SimpleMLP:
 [[464  85 357]
 [122 281 281]
 [268 186 956]]

 Classification Report for SimpleMLP:
              precision    recall  f1-score   support

 alpha-helix       0.54      0.51      0.53       906
  beta-sheet       0.51      0.41      0.45       684
        coil       0.60      0.68      0.64      1410

    accuracy                           0.57      3000
   macro avg       0.55      0.53      0.54      3000
weighted avg       0.56      0.57      0.56      3000

Confussion Matrix for AdvanceMLP:
 [[390 163 239]
 [164 430 294]
 [281 274 765]]

 Classification Report AdvanceMLP:
              precision    recall  f1-score   support

 alpha-helix       0.47      0.49      0.48       792
  beta-sheet       0.50      0.48      0.49       888
        coil       0.59      0.58      0.58      1320

    accuracy                           0.53      3000
   macro avg       0.52      0.52      0.52      3000
weighted avg       0.53      0.53      0.53      3000


"""