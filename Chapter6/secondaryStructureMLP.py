import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
import random

amino_acids = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
#print(len(amino_acids))
alphabet_size = len(amino_acids)+1

def generate_sample_sequences(num_sequences, sequence_length):
    sequences = []
    for _ in range(num_sequences):
        sequence = np.random.choice(amino_acids, sequence_length).tolist()
        sequences.append(sequence)
    return sequences

def create_orthogonal_encoding(amino_acids,alphabet_size):
    encoding_map = {}
    for i,aa in enumerate(amino_acids):
        encoding = np.zeros(alphabet_size)
        encoding[i] = 1
        encoding_map[aa] = encoding
    encoding_map['<TERMINATOR>'] = np.zeros(alphabet_size)
    encoding_map['<TERMINATOR>'][-1] = 1
    return encoding_map

encoding_map = create_orthogonal_encoding(amino_acids,alphabet_size)
#print(f"Encoding map: {encoding_map}")

def encode_sequence(sequence,encoding_map,window_size):
    padded_sequence = ['<TERMINATOR>'] * ((window_size - 1) // 2) + sequence + ['<TERMINATOR>'] * ((window_size - 1) // 2)
    encoded_sequence = []
    for i in range(len(padded_sequence) - window_size + 1):
        window = padded_sequence[i:i + window_size]
        encoded_window = [encoding_map[residue] for residue in window]
        encoded_sequence.append(np.concatenate(encoded_window, axis=0))
    return np.array(encoded_sequence)

num_sequences = 500
sequence_length = 30
window_size = 13

sequences = generate_sample_sequences(num_sequences, sequence_length)

encoded_sequences = []
for seq in sequences:
    encoded_seq = encode_sequence(seq,encoding_map,window_size)
    encoded_sequences.append(encoded_seq)
    #print(f"Encoded seq: {encoded_seq}")

#print("Example encoded sequence shape:", len(encoded_sequences[1]), "x", encoded_sequences[0].shape[1])