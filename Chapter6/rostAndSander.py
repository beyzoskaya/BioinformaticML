import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
alphabet_size = len(amino_acids)

def create_one_hot_encoding(amino_acids):
    encoding_map = {aa: i for i, aa in enumerate(amino_acids)}
    encoding_map['<TERMINATOR>'] = len(amino_acids)
    return encoding_map

encoding_map = create_one_hot_encoding(amino_acids)

def generate_sample_sequences(num_sequences, sequence_length):
    sequences = []
    for _ in range(num_sequences):
        sequence = np.random.choice(amino_acids, sequence_length).tolist()
        sequences.append(sequence)
    return sequences

def generate_profiles_with_local_encoding(sequences, window_size, encoding_map):
    profiles = []
    for seq in sequences:
        padded_sequence = ['<TERMINATOR>'] * ((window_size - 1) // 2) + seq + ['<TERMINATOR>'] * ((window_size - 1) // 2)
        for i in range(len(padded_sequence) - window_size + 1):
            window = padded_sequence[i:i+window_size]
            profile = np.zeros((window_size, len(encoding_map)))

            for j, aa in enumerate(window):
                if aa in encoding_map:
                    profile[j, encoding_map[aa]] = 1

            profiles.append(profile)
    
    return np.array(profiles)

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

class RiisKroghNet(nn.Module):
    def __init__(self, input_size, window_size, num_classes):
        super(RiisKroghNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=20, kernel_size=3, stride=1, padding=1)  # 20 filters
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)  # Pooling with period of 3 residues

        #conv_output_size = (window_size - 1) // 3 + 1

        #print(f'Calculated conv_output_size: {conv_output_size}') 

        self.fc1 = nn.Linear(20 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #print(f'Input shape: {x.shape}')
        x = x.permute(0, 2, 1)  # (batch_size, num_features, window_size)
        #print(f'After permute: {x.shape}')
        x = self.relu(self.conv1(x))
        #print(f'After conv1: {x.shape}')
        x = self.pool(x)
        #print(f'After pooling: {x.shape}')
        x = x.flatten(start_dim=1)
        #print(f'After flatten: {x.shape}')
        x = self.relu(self.fc1(x))
        #print(f'After fc1: {x.shape}')
        x = self.fc2(x)
        #print(f'Output shape: {x.shape}')
        return x


num_sequences = 500
sequence_length = 30
window_size = 13
num_classes = 3 

sequences = generate_sample_sequences(num_sequences, sequence_length)
profiles_with_local_encoding = generate_profiles_with_local_encoding(sequences, window_size, encoding_map)
labels = encode_labels(sequences, window_size)

# Profiles with local encoding to have shape (num_samples, window_size, num_features)
X_train, X_test, y_train, y_test = train_test_split(profiles_with_local_encoding, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

patience = 5  # Number of epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

ensemble_size = 5
models = []

for ensemble_idx in range(ensemble_size):
    print(f"Training ensemble model {ensemble_idx + 1}")
    
    model = RiisKroghNet(input_size=len(encoding_map), window_size=window_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Ensemble {ensemble_idx + 1}, Epoch [{epoch + 1}/20], Loss: {epoch_loss:.4f}')
        

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(test_loader.dataset)
        print(f'Ensemble {ensemble_idx + 1}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping for ensemble model {ensemble_idx + 1}")
                break
    
    models.append(model)

def ensemble_predict(models, dataloader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            batch_preds = np.mean([torch.softmax(model(inputs), dim=1).cpu().numpy() for model in models], axis=0)
            all_preds.extend(np.argmax(batch_preds, axis=1))
    return np.array(all_preds)

y_test_preds = ensemble_predict(models, test_loader)
accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_test_preds)
print(f'Ensemble Test Accuracy: {accuracy:.4f}')


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

for i, model in enumerate(models):
    accuracy, _, _ = evaluate_model(model, test_loader)
    print(f"Model {i+1} Test Accuracy: {accuracy:.4f}")

"""

Model 1 Test Accuracy: 0.9500
Model 2 Test Accuracy: 0.8747
Model 3 Test Accuracy: 0.8573
Model 4 Test Accuracy: 0.8527
Model 5 Test Accuracy: 0.8507

"""

def print_dataset_parts(X, y, dataset_name):
    print(f"{dataset_name} Data Samples:")
    for i in range(len(X)):
        print(f"Profile {i + 1}:")
        print(f"Features: {X[i]}")
        print(f"Label: {y[i]}")
        print()

#print_dataset_parts(X_train, y_train, "Training")
#print_dataset_parts(X_test, y_test, "Testing")
