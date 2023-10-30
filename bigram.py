import math
from transcribe import Transcriber
from collections import defaultdict
from nltk.tokenize import word_tokenize
import numpy


DATASETS = [
	("datasets\\train\\bos", "bos"),
	("datasets\\train\\hrv", "hrv"),
	("datasets\\train\\srp", "srp"),
]


class BigramClassifier:

	
	def __init__(self) -> None:

		dataset = {}

		for path, label in DATASETS:

			with open(path, "r", encoding="utf-8") as file:

				for line in file:
					
					try:

						tokens = line.strip().split(" ")
						tokens = ["[S]"] + tokens + ["[E]"]
						
						for index in range(len(tokens) - 1):
							
							token_a = tokens[index]
							token_b = tokens[index + 1]

							dataset.setdefault(label, {})
							dataset[label].setdefault(token_a, {})
							dataset[label][token_a].setdefault(token_b, 0)
							dataset[label][token_a][token_b] += 1

					except Exception as e:

						print(f"{e}")

		self.dataset

		for label in dataset:

			for token_a in dataset[label]:

				total_frequency = sum(token_a.values())

				for token_b in token_a:

					dataset.setdefault(label, defaultdict(lambda: defaultdict))
					dataset[label].setdefault(token_a, {})
					dataset[label][token_a].setdefault(token_b, 0)
					dataset[label][token_a][token_b] += 1


	def classify(self, text):

		labels = []
		probabilities = []

		text = word_tokenize(Transcriber().transcribe(text.lower(), output="latin"))
		print(" ".join(text))

		text = ["[S]"] + text + ["[E]"]

		for label in self.dataset:		
			labels.append(label)
			probabilities.append(sum([self.dataset[label][text[index]][text[index + 1]] for index in range(len(text) - 1)]))

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
		print()

		return labels[numpy.argmax(normalized_probabilities)]

		
if __name__ == "__main__":

	result = BigramClassifier().classify("ja videću ga")

	#print(f"Classified as: {result.upper()}")