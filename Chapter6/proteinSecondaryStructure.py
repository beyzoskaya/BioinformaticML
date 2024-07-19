import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

print(f"Number of protein sequences: {len(protein_sequences)}")
labels = [1 if i % 2 == 0 else 0 for i in range(len(protein_sequences))]
print(f"Number of labels: {len(labels)}")

encoded_sequences = encode_protein_sequences(protein_sequences)

max_length = max(len(seq) for seq in encoded_sequences)
padded_encoded_sequences = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant') for seq in encoded_sequences])

X_train, X_test, y_train, y_test = train_test_split(padded_encoded_sequences, labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

input_size = X_train_tensor.shape[1] * X_train_tensor.shape[2]
model = SimpleNN(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_binary = (y_pred > 0.5).float()
    print(classification_report(y_test_tensor.numpy(), y_pred_binary.numpy()))