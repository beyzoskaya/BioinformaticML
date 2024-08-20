import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_sequence(length, signal_peptide_length, cleavage_site_position):
    sequence = np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'),length)
    signal_peptide = sequence[:signal_peptide_length]
    mature_protein = sequence[signal_peptide_length:]

    labels = np.zeros(length)
    labels[:signal_peptide_length] = 1

    return sequence,labels, cleavage_site_position

seq_length = 50
signal_peptide_length = 20
cleavage_site_position = 20

sequence, labels, cleavage_site_position = generate_synthetic_sequence(seq_length, signal_peptide_length,cleavage_site_position)

print(f"Sequence: {''.join(sequence)}")
print(f"Labels: {labels}")

def calculate_s_score(sequence):
    return np.linspace(1,0,len(sequence))

def calculate_c_score(sequence,cleavage_site_position):
    scores = np.zeros(len(sequence))
    scores[cleavage_site_position] = 1.0
    return scores

s_scores = calculate_s_score(sequence)
c_scores = calculate_c_score(sequence, cleavage_site_position)
print(f"s_score: {s_scores}")
print(f"c_score: {c_scores}")

def calculate_y_score(c_scores,s_scores, d=2):
    delta_s_scores = np.zeros_like(s_scores)
    for i in range(d, len(s_scores) - d):
        delta_s_scores[i] = np.mean(s_scores[i-d:i]) - np.mean(s_scores[i+1:i+1+d])
    
    y_scores = c_scores * delta_s_scores
    return y_scores

y_scores = calculate_y_score(c_scores, s_scores)
print(f"y_score: {y_scores}")

plt.figure(figsize=(10,6))
plt.subplot(3, 1, 1)
plt.plot(s_scores, label='S-score', color='blue')
plt.title('S-score (Signal Peptide Probability)')
plt.ylabel('Score')
plt.xlabel('Position')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(c_scores, label='C-score', color='green')
plt.title('C-score (Cleavage Site Probability)')
plt.ylabel('Score')
plt.xlabel('Position')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(y_scores, label='Y-score', color='red')
plt.title('Y-score (Combined Score)')
plt.ylabel('Score')
plt.xlabel('Position')
plt.legend()

plt.tight_layout()
plt.savefig('signalIP.png')
plt.show()