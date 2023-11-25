from unigram import UnigramClassifier
from bigram import BigramClassifier
import math
import numpy

class CombinedUnigramBigramClassifier:


	def __init__(self):

		self.unigram_classifier = UnigramClassifier()
		self.bigram_classifier = BigramClassifier()


	def classify(self, text, only_probabilities=False):

		p = {}

		unigram_probabilities = self.unigram_classifier.classify(text, only_probabilities=True)
		bigram_probabilities = self.bigram_classifier.classify(text, only_probabilities=True)

		for l in [unigram_probabilities, bigram_probabilities]:
			for label, probability in l:
				p.setdefault(label, 0.0)
				p[label] += probability

		labels = []
		probabilities = []

		for label, probability in p.items():
			labels.append(label)
			probabilities.append(probability)

		if only_probabilities:
			return list(zip(labels, probabilities))

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

	classifier = CombinedUnigramBigramClassifier()

	classifier.classify("ja videću ga")
	classifier.classify("ja vidjeću ga")
	classifier.classify("ja videt ću ga")
	classifier.classify("ja vidjet ću ga")

	#print(f"Classified as: {result.upper()}")