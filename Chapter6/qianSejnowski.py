import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Bio.Align import PairwiseAligner
from sklearn.model_selection import train_test_split

amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
alphabet_size = len(amino_acids)

def generate_profiles_with_indels(sequences, window_size, encoding_map):
    profiles = []
    aligner = PairwiseAligner()
    for seq in sequences:
        padded_sequence = ['<TERMINATOR>'] * ((window_size - 1) // 2) + seq + ['<TERMINATOR>'] * ((window_size - 1) // 2)
        #print(f"Padded sequence: {padded_sequence}")
        for i in range(len(padded_sequence) - window_size + 1):
            window = padded_sequence[i:i+window_size]
            profile = np.zeros((window_size, len(encoding_map)))

            window_str = ''.join(window)
            alignments = aligner.align(window_str, window_str)
            best_alignment = alignments[0]
            aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
            num_deletions = aligned_seq1.count('-')
            num_insertions = aligned_seq2.count('-')

            for j, aa in enumerate(window):
                if aa in encoding_map:
                    profile[j, encoding_map[aa]] = 1

            conservation_weight = np.mean([encoding_map.get(aa, 0) for aa in window if aa in encoding_map])
            profile = np.hstack([profile.flatten(), [num_deletions, num_insertions, conservation_weight]])
            profiles.append(profile)
            #for profile in profiles:
                #print(f"Profile: {profile}")
    return np.array(profiles)

class QianSejnowskiNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(QianSejnowskiNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.5)  # Dropout with 50% probability
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def encode_labels(sequences, window_size):
    alpha_helix_aa = {'Ala', 'Leu', 'Met', 'Glu', 'Lys'}
    beta_sheet_aa = {'Val', 'Ile', 'Phe', 'Tyr', 'Cys'}
    coil_aa = {'Gly', 'Pro', 'Ser', 'Asn', 'Gln'}
    
    labels = []
    for seq in sequences:
        padded_sequence = ['<TERMINATOR>'] * ((window_size - 1) // 2) + seq + ['<TERMINATOR>'] * ((window_size - 1) // 2)
        for i in range(len(padded_sequence) - window_size + 1):
            window = padded_sequence[i:i+window_size]
            alpha_count = sum(aa in alpha_helix_aa for aa in window)
            beta_count = sum(aa in beta_sheet_aa for aa in window)
            coil_count = sum(aa in coil_aa for aa in window)
            
            if alpha_count > beta_count and alpha_count > coil_count:
                labels.append(0)  # alpha-helix
            elif beta_count > alpha_count and beta_count > coil_count:
                labels.append(1)  # beta-sheet
            else:
                labels.append(2)  # coil
    
    return np.array(labels)

def generate_sample_sequences(num_sequences, sequence_length):
    sequences = []
    for _ in range(num_sequences):
        sequence = np.random.choice(amino_acids, sequence_length).tolist()
        sequences.append(sequence)
    return sequences

def create_one_hot_encoding(amino_acids):
    encoding_map = {aa: i for i, aa in enumerate(amino_acids)}
    encoding_map['<TERMINATOR>'] = len(amino_acids)
    encoding_map['<INSERTION>'] = len(amino_acids) + 1
    return encoding_map

num_sequences = 500
sequence_length = 30
window_size = 13
hidden_size1 = 128
hidden_size2 = 64
num_classes = 3  

sequences = generate_sample_sequences(num_sequences, sequence_length)
encoding_map = create_one_hot_encoding(amino_acids)


profiles_with_indels = generate_profiles_with_indels(sequences, window_size, encoding_map)
labels = encode_labels(sequences, window_size)

X_train, X_test, y_train, y_test = train_test_split(profiles_with_indels, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = profiles_with_indels.shape[1]
model = QianSejnowskiNet(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean()
        return accuracy.item()

accuracy = evaluate_model(model, X_test_tensor, y_test_tensor)
print(f'Test Accuracy: {accuracy:.4f}')

def print_dataset_parts(X, y, dataset_name):
    print(f"{dataset_name} Data Samples:")
    for i in range(len(X)):
        print(f"Profile {i + 1}:")
        print(f"Features: {X[i]}")
        print(f"Label: {y[i]}")
        print()

#print_dataset_parts(X_train, y_train, "Training")
#print_dataset_parts(X_test, y_test, "Testing")

"""
Each window labeled separately
Each window converted into a profile that includes aa info and additional features like insertions and deletions
Each profile has:
    one hot encoding of aa
    count of insertions and deletions
    conservation weight
"""