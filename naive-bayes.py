import os
from collections import defaultdict
import math

SENTENCE = "ja želim da slušam srpsku muziku"

PROBABILITY_MATRICES_PATH = "probability-matrices"

probability_matrices = []

for filename in os.listdir(PROBABILITY_MATRICES_PATH):

	with open(os.path.join(PROBABILITY_MATRICES_PATH, filename), "r", encoding="utf-8") as file:

		probability_matrix = None

		for line in file:

			type, probability = line.split("	")

			probability = float(probability)

			if probability_matrix is None:

				probability_matrix = defaultdict(lambda: probability)

			else:

				probability_matrix[type] = probability

	probability_matrices.append((filename, probability_matrix))

labels = []
probabilities = []

for label, probability_matrix in probability_matrices:
	labels.append(label)
	probabilities.append(sum([probability_matrix[token] for token in SENTENCE.split(" ")]))

probability_max = max(probabilities)

normalized_probabilities = []

for label, probability in zip(labels, probabilities):
	p = math.exp(probability-probability_max)
	normalized_probabilities.append(p)

for n in range(len(normalized_probabilities)):
	p = normalized_probabilities[n]
	p = p / sum(normalized_probabilities)
	normalized_probabilities[n] = p

for label, normalized_probability in zip(labels, normalized_probabilities):
	print(f"{label}: {int(normalized_probability * 100)}%")