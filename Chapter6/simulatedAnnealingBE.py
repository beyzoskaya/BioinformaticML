import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random
import math

def binary_encoding(sequences):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding_map = {aa: [int(i == j) for i in range(len(amino_acids))] for j, aa in enumerate(amino_acids)}

    encoded_sequences = []
    for seq in sequences:
        encoded_seq = []
        for amino_acid in seq:
            encoded_seq.append(encoding_map.get(amino_acid, [0] * len(amino_acids)))
        encoded_sequences.append(encoded_seq)
    return encoded_sequences, len(amino_acids)

def pad_sequences(encoded_sequences):
    encoded_sequences_np = [np.array(seq) for seq in encoded_sequences]
    max_length = max(seq.shape[0] for seq in encoded_sequences_np)
    padded_sequences = np.array([np.pad(seq, ((0, max_length - seq.shape[0]), (0, 0)), mode='constant') for seq in encoded_sequences_np])
    #for seq in padded_sequences:
    #    print(f"Length of padded sequence: {len(seq)}")
    return padded_sequences

"""
Example encoding

Protein Sequence 84:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

"""
"""
data shape: (68,53,20)
68 number of sequences
53 padded sequence length
20 size of one hot encoding for each amino acid
batch size 32 --> 2 full batches and 1 partial batch 32x2 + 4 = 68

flattened input: (32,53,20) to (32,1060)

"""

class SimpleNN(nn.Module):
    def __init__(self,input_size):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size,64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MoreSimpleNN(nn.Module):
    def __init__(self, input_size):
        super(MoreSimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train(model, criterion, optimizer, train_loader, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs,labels in train_loader:
            inputs,labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return losses

def evaluate(model,test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs,labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.round().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(predicted)
    print(classification_report(y_true, y_pred))

def plot_loss_curve(losses_default, losses_optimized):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(losses_default) + 1), losses_default, marker='o', label='Default Hyperparameters Loss')
    plt.plot(range(1, len(losses_optimized) + 1), losses_optimized, marker='x', label='Optimized Hyperparameters Loss')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_optimized_sa.png')
    plt.show()


def print_nn_dimensions(model):
    print("\nNeural Network Dimensions:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")

def simulated_annealing(encoded_sequences, labels, max_iterations=100, initial_temp=1000, cooling_rate=0.95):
    def evaluate_hyperparameters(hidden_size, learning_rate):
        X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=0.2, random_state=42)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=list(zip(X_test_tensor, y_test_tensor)), batch_size=32, shuffle=False)
        
        model = SimpleNN(X_train_tensor.shape[1] * X_train_tensor.shape[2])
        model.fc1 = nn.Linear(model.fc1.in_features, hidden_size)
        model.fc2 = nn.Linear(hidden_size, 1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        num_epochs = 10
        losses = train(model, criterion, optimizer, train_loader, num_epochs)
        return losses[-1]
    
    best_params = {'hidden_size': 64, 'learning_rate': 0.001}
    best_loss = float('inf')
    current_temp = initial_temp

    for iteration in range(max_iterations):
        hidden_size = random.choice([16, 32, 64])
        learning_rate = 10**random.uniform(-3, -1)
        
        loss = evaluate_hyperparameters(hidden_size, learning_rate)
        if loss < best_loss:
            best_loss = loss
            best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate}
        if random.uniform(0, 1) < math.exp((best_loss - loss) / current_temp):
            best_params = {'hidden_size': hidden_size, 'learning_rate': learning_rate}
        current_temp *= cooling_rate
    
    print(f"Final Best Params: {best_params}") 
    return best_params


protein_sequences = [
    "MKTAYIAKQRQISFVKSHFSRQDILDLWQEKNRLAAHPPFASWRNSEEARTDR",
    "MKQLQETNKAVMAAFTGQFAGDDAPRAVFPSIVGRPRHQGVMVGMGQDS",
    "MTYKLTGRSGSDVVVVSMEGSSSMDSIKKFGKIKRDIDGKLMVTY",
    "MQVGTIKVMGIQLYGVNGILVRTKKSMGGQSKKKADEQVRLKKDL",
    "MDSYKLTTKISQGQGLSLLSILDGQVDGQSIKAFVKHDLKVLSD",
    "MAARGLTVAVSAVGIAYFALPPEQQGAGNTGGAGDEKTGAAHSDD",
    "MGTSKQWLGSMKYDPHGDHGNSVTLSEWVNKTKLQSIEDGFQ",
    "MNPADLNNGNSHQQVRPFTIRFDKVLNTKVQGYSHVQKFLSQEE",
    "MVVGLKDLQSALKEEKVLLTQGNLLHISGKLAQEFGPVLPKGH",
    "MATLKGAAGGSFSYGVLGIAYTGRFEPRADSVPTAPLGGIPQT",
    "MNTFVTADSGQVVVTVGQELTGKIKKFGSLVRTTDNKLMVTY",
    "MPEAKNQNGKRVLEDFLVGFYTREKSYIKTVEGSDDKRLEAQ",
    "MATPGAPGTSNTGNTPGKRGGSFSYGFLSRYPESAEISLVD",
    "MGGSFSFGVLGYFRYPESPKKVNVLDNFGNLPQNLKPK",
    "MGRRGGFSYGFLSRYPRSAPVILKGGQSKKTADEQVR",
    "MKTAIAKQRQISFVKSHFSRQDILDLWQEKNRLAA",
    "MNSSKQWLGSMKYDPHGDHGNSVTLSEWVNKTKL",
    "MATGKQSPGKSQSGSFSYGFLSRYPESAEISLV",
    "MASTGFITKISQGQGLSLLSILDGQVDGQSIKAF",
    "MKLSPGSSRGSDVVVVSMEGSSSMDSIKKFGKI",
    "MGGFSYGVLGIAYTGRFEPRADSVPTAPLGGIPQT",
    "MGRQSPGKSQSGSFSYGFLSRYPESAEISLV",
    "MKTAIAKQRQISFVKSHFSRQDILDLWQEKNR",
    "MGTYKLTGRSGSDVVVVSMEGSSSMDSIKK",
    "MNSSKQWLGSMKYDPHGDHGNSVTLSEWVNK",
    "MATGKQSPGKSQSGSFSYGFLSRYPESAEI",
    "MDSYKLTTKISQGQGLSLLSILDGQVDGQ",
    "MKQLQETNKAVMAAFTGQFAGDDAPRAV",
    "MATPGAPGTSNTGNTPGKRGGSFSYGF",
    "MATLKGAAGGSFSYGVLGIAYTGRFE",
    "MPEAKNQNGKRVLEDFLVGFYTREKS",
    "MGGFSYGVLGIAYTGRFEPRADSVPT",
    "MGRRGGFSYGFLSRYPRSAPVILKG",
    "MKTAIAKQRQISFVKSHFSRQDIL",
    "MNSSKQWLGSMKYDPHGDHGNSVTL",
    "MATGKQSPGKSQSGSFSYGFLSRY",
    "MASTGFITKISQGQGLSLLSILD",
    "MKLSPGSSRGSDVVVVSMEGSS",
    "MGTYKLTGRSGSDVVVVSME",
    "MKQLQETNKAVMAAFTGQ",
    "MGTSKQWLGSMKYDPH",
    "MNPADLNNGNSHQQVR",
    "MVVGLKDLQSALKEE",
    "MATLKGAAGGSFSY",
    "MNTFVTADSGQVV",
    "MPEAKNQNGKRV",
    "MATPGAPGTSN",
    "MGGSFSFGVL",
    "MGGRRQSPGK",
    "MGGFSYGVL",
    "MGRRGGFSY",
    "MKTAIAKQR",
    "MNSSKQWL",
    "MATGKQSP",
    "MASTGFIT",
    "MKLSPGSS",
    "MGTYKLTG",
    "MKQLQETN",
    "MGTSKQWL",
    "MNPADLNN",
    "MVVGLKDL",
    "MATLKGA",
    "MNTFVTA",
    "MPEAKNQ",
    "MATPGAP",
    "MGGSFSF",
    "MGGRRQ",
    "MGGFSY",
    "MGRRGG",
    "MKTAIA",
    "MNSSKQ",
    "MATGKQ",
    "MASTGF",
    "MKLSPG",
    "MGTYKL",
    "MKQLQE",
    "MGTSKQ",
    "MNPADL",
    "MVVGLK",
    "MATLKG",
    "MNTFVT",
    "MPEAKN",
    "MATPGA",
    "MGGSFS",
    "MGGRRQ"
]
labels = [1 if i % 2 == 0 else 0 for i in range(len(protein_sequences))]
encoded_sequences, num_of_aa = binary_encoding(protein_sequences)
print(f"Number of amino acids: {num_of_aa}")
#for seq in encoded_sequences:
#    print(np.array(seq).shape)

padded_encoded_sequences = pad_sequences(encoded_sequences)

encoded_sequences_train, encoded_sequences_test, labels_train, labels_test = train_test_split(padded_encoded_sequences, labels, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(padded_encoded_sequences, labels, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=list(zip(X_test_tensor, y_test_tensor)), batch_size=32, shuffle=False)

print("Running simulated annealing for hyperparameter optimization...")
best_params = simulated_annealing(padded_encoded_sequences, labels)
print(f"Best Hyperparameters from Simulated Annealing: {best_params}")

optimized_hidden_size = best_params['hidden_size']
optimized_learning_rate = best_params['learning_rate']
input_size = 53*20
model_optimized = SimpleNN(input_size)
model_optimized.fc1 = nn.Linear(model_optimized.fc1.in_features, optimized_hidden_size)
model_optimized.fc2 = nn.Linear(optimized_hidden_size, 1)
optimizer_optimized = optim.Adam(model_optimized.parameters(), lr=optimized_learning_rate)

print("Training with Optimized Hyperparameters...")
losses_optimized = train(model_optimized, nn.BCELoss(), optimizer_optimized, train_loader, num_epochs=10)
print("Evaluating Optimized Model...")
evaluate(model_optimized, test_loader)

model_default = SimpleNN(input_size)
optimizer_default = optim.Adam(model_default.parameters(), lr=0.001)

print("Training with Default Hyperparameters...")
losses_default = train(model_default, nn.BCELoss(), optimizer_default, train_loader, num_epochs=10)
print("Evaluating Default Model...")
evaluate(model_default, test_loader)
plot_loss_curve(losses_default, losses_optimized)

# train_loader = torch.utils.data.DataLoader(dataset=list(zip(torch.tensor(encoded_sequences_train, dtype=torch.float32),
#                                                             torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1))),
#                                            batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=list(zip(torch.tensor(encoded_sequences_test, dtype=torch.float32),
#                                                            torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1))),
#                                           batch_size=32, shuffle=False)
# print(f"Shape of encoded_sequences_train: {encoded_sequences_train.shape}")
# input_size = 53*20
# model_16 = MoreSimpleNN(input_size)
# model_64 = SimpleNN(input_size)

# criterion = nn.BCELoss()
# optimizer_16 = optim.Adam(model_16.parameters(), lr=0.001)
# optimizer_64 = optim.Adam(model_64.parameters(), lr=0.001)

# print("Training MoreSimpleNN...")
# losses_16 = train(model_16, criterion, optimizer_16, train_loader, num_epochs=10)
# print("Evaluating MoreSimpleNN...")
# evaluate(model_16, test_loader)

# print("Training SimpleNN...")
# losses_64 = train(model_64, criterion, optimizer_64, train_loader, num_epochs=10)
# print("Evaluating SimpleNN...")
# evaluate(model_64, test_loader)

# print_nn_dimensions(model_16)
# print_nn_dimensions(model_64)

# plot_loss_curve(losses_16, losses_64)

#print("\nEncoded Protein Sequences:")
#for i,encoded_seq in enumerate(encoded_sequences):
#    print(f"\nProtein Sequence {i + 1}:")
#    for encoded_aa in encoded_seq:
#        print(encoded_aa)