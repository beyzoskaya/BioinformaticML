import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def encode_protein_sequences(sequences):
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = []
        for amino_acid in seq:
            if amino_acid in ['A', 'G', 'V', 'L', 'I']:
                encoded_seq.append([1, 0, 0, 0])  # hydrophobic amino acids
            elif amino_acid in ['R', 'K', 'D', 'E']:
                encoded_seq.append([0, 1, 0, 0])  # charged amino acids
            else:
                encoded_seq.append([0, 0, 1, 0])  # other amino acids
        encoded_sequences.append(encoded_seq)
    return encoded_sequences

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class moreSimpleNN(nn.Module):
    def __init__(self, input_size):
        super(moreSimpleNN, self).__init__()
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

def train(model, criterion, optimizer, train_loader, num_epochs):
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return losses

def plot_loss_curve(losses_16, losses_64):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses_16) + 1), losses_16, marker='o', label='moreSimpleNN')
    plt.plot(range(1, len(losses_64) + 1), losses_64, marker='o', label='SimpleNN')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot_different_NN.png')
    plt.show()


def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs.round().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(predicted)
    print(classification_report(y_true, y_pred))

#print(f"Number of protein sequences: {len(protein_sequences)}")
labels = [1 if i % 2 == 0 else 0 for i in range(len(protein_sequences))]

encoded_sequences = encode_protein_sequences(protein_sequences)
max_length = max(len(seq) for seq in encoded_sequences)
padded_encoded_sequences = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant') for seq in encoded_sequences])

X_train, X_test, y_train, y_test = train_test_split(padded_encoded_sequences, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # [32] --> [32,1]
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)   # [32] --> [32,1]

train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=list(zip(X_test_tensor, y_test_tensor)), batch_size=32, shuffle=False)

input_size = X_train_tensor.shape[1] * X_train_tensor.shape[2]
model_16 = moreSimpleNN(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_16.parameters(), lr=0.001)
num_epochs = 10
losses_16 = train(model_16, criterion, optimizer, train_loader, num_epochs)


model_64 = SimpleNN(input_size)
optimizer_64 = optim.Adam(model_64.parameters(), lr=0.001)
losses_64 = train(model_64, criterion, optimizer_64, train_loader, num_epochs)

plot_loss_curve(losses_16, losses_64)
evaluate(model_16, test_loader)
evaluate(model_64, test_loader)

#input_size = X_train_tensor.shape[1] * X_train_tensor.shape[2]
#model = SimpleNN(input_size)
#criterion = nn.BCELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001)

#num_epochs = 10
#batch_size = 32