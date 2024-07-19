import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from qianSejnowski import generate_sample_sequences

# Define amino acids and encoding
amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
alphabet_size = len(amino_acids)

def create_one_hot_encoding(amino_acids):
    encoding_map = {aa: i for i, aa in enumerate(amino_acids)}
    encoding_map['<TERMINATOR>'] = len(amino_acids)
    return encoding_map

encoding_map = create_one_hot_encoding(amino_acids)

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
    def __init__(self, input_size, window_size, num_classes, M, hidden_units):
        super(RiisKroghNet, self).__init__()
        self.M = M
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=M, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        conv_output_size = (window_size - 1) // 3 + 1
        self.fc1 = nn.Linear(M * conv_output_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_size, window_size)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Sample data generation
num_sequences = 500
sequence_length = 30
window_size = 15
num_classes = 3
M = 3
sequences = generate_sample_sequences(num_sequences, sequence_length)
profiles_with_local_encoding = generate_profiles_with_local_encoding(sequences, window_size, encoding_map)
labels = encode_labels(sequences, window_size)

X_train, X_test, y_train, y_test = train_test_split(profiles_with_local_encoding, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

hidden_units_list = [16, 32, 64, 128, 256]
models = []
for hidden_units in hidden_units_list:
    model = RiisKroghNet(input_size=len(encoding_map), window_size=window_size, num_classes=num_classes, M=M, hidden_units=hidden_units)
    models.append(model)

def train_model(model, train_loader, test_loader, num_epochs=20, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(test_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    return model

# Train all models
trained_models = []
for model in models:
    print(f"Training model with {model.fc1.out_features} hidden units...")
    trained_model = train_model(model, train_loader, test_loader)
    trained_models.append(trained_model)

def ensemble_predict(models, dataloader):
    model_preds = [model.eval() for model in models]
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            ensemble_outputs = []
            for model in model_preds:
                outputs = model(inputs)
                ensemble_outputs.append(outputs)
            ensemble_outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)
            preds = torch.argmax(ensemble_outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

y_test_preds = ensemble_predict(trained_models, test_loader)
accuracy = accuracy_score(y_test_tensor.cpu().numpy(), y_test_preds)
print(f'Ensemble Test Accuracy: {accuracy:.4f}')
