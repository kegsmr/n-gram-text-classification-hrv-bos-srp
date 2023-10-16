import os
from collections import defaultdict
import math
from nltk.tokenize import word_tokenize
from transcribe import Transcriber
import numpy

def classify(text):

	PROBABILITY_MATRICES_PATH = "probability-matrices"

	probability_matrices = []

	for filename in os.listdir(PROBABILITY_MATRICES_PATH):

		with open(os.path.join(PROBABILITY_MATRICES_PATH, filename), "r", encoding="utf-8") as file:

			first_line = True

			probability_matrix = None

			for line in file:

				if first_line:

					type_count, word_count = line.split("	")

					type_count = int(type_count)
					word_count = int(word_count)

					first_line = False

				else:

					type, probability = line.split("	")

					probability = float(probability)

					if probability_matrix is None:

						probability_matrix = defaultdict(lambda: probability)

					else:

						probability_matrix[type] = probability

		probability_matrices.append((filename, type_count, word_count, probability_matrix))

	total_type_count = sum([item[1] for item in probability_matrices])
	total_word_count = sum([item[2] for item in probability_matrices])

	labels = []
	probabilities = []

	text = word_tokenize(Transcriber().transcribe(text.lower(), output="latin"))
	print(" ".join(text))

	for label, type_count, word_count, probability_matrix in probability_matrices:
		labels.append(label)
		probabilities.append(sum([probability_matrix[token] for token in text]) + math.log((word_count + type_count) / (total_word_count + total_type_count)))

	probability_max = max(probabilities)

	normalized_probabilities = []

	for label, probability in zip(labels, probabilities):
		p = math.exp(probability-probability_max)
		normalized_probabilities.append(p)

	normalized_probabilities_sum = sum(normalized_probabilities)

	for n in range(len(normalized_probabilities)):
		p = normalized_probabilities[n]
		p = p / normalized_probabilities_sum
		normalized_probabilities[n] = p

	for label, normalized_probability in zip(labels, normalized_probabilities):
		print(f"{label.upper()}: {round(normalized_probability * 100)}%")

	return labels[numpy.argmax(normalized_probabilities)]

if __name__ == "__main__":
	result = classify(open("naive-bayes-input.txt", "r", encoding="utf-8").read())
	print(f"Classified as: {result.upper()}")