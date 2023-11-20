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

SENTENCE_START = "[S]"
SENTENCE_END = "[E]"
UNKNOWN_WORD = "<UNK>"


class BigramClassifier:

	
	def __init__(self) -> None:

		d = {}

		for path, label in DATASETS:

			d.setdefault(label, {})
			d[label].setdefault(UNKNOWN_WORD, {})
			d[label][UNKNOWN_WORD].setdefault(UNKNOWN_WORD, 0.0)

			with open(path, "r", encoding="utf-8") as file:

				for line in file:
					
					try:

						tokens = line.strip().split(" ")
						tokens = [SENTENCE_START] + tokens + [SENTENCE_END]
						
						for index in range(len(tokens) - 1):
							
							a_token = tokens[index]
							b_token = tokens[index + 1]
							
							d[label].setdefault(a_token, {})
							d[label][a_token].setdefault(b_token, 0.0)
							d[label][a_token][b_token] += 1.0

							d[label][a_token].setdefault(UNKNOWN_WORD, 0.0)

					except Exception as e:

						print(f"{e}")

		self.dataset = d

		"""d = {}

		for label, a_tokens in self.dataset.items():

			d.setdefault(label, {})
			d[label].setdefault(UNKNOWN_WORD, {})
			d[label][UNKNOWN_WORD].setdefault(UNKNOWN_WORD, 0.0)

			for a_token, b_tokens in a_tokens.items():

				#if sum(b_tokens.values()) == 1:
				#	a_token = UNKNOWN_WORD

				d[label].setdefault(a_token, {})
				d[label][a_token].setdefault(UNKNOWN_WORD, 0.0)

				for b_token, frequency in b_tokens.items():

					#if frequency == 1:
					#s	b_token = UNKNOWN_WORD 

					d[label][a_token].setdefault(b_token, 0.0)
					d[label][a_token][b_token] += frequency

		self.dataset = d"""

		d = {}

		for label, a_tokens in self.dataset.items():
			d.setdefault(label, {})
			for a_token, b_tokens in a_tokens.items():
				d[label].setdefault(a_token, {})
				for b_token, frequency in b_tokens.items():
					d[label][a_token][b_token] = math.log((1.0 + frequency) / (1.0 + len(d[label][a_token].values()) + sum(d[label][a_token].values())))

		self.dataset = d

		"""for label, a_tokens in self.dataset.items():
			for a_token, b_tokens in a_tokens.items():
				for b_token, frequency in b_tokens.items():
					print(f"{label}	{a_token}	{b_token}	{frequency}")"""


	def dump(self):
		with open("bigram_dump.txt", "w", encoding="utf-8") as file:
			for label, a_tokens in self.dataset.items():
				for a_token, b_tokens in a_tokens.items():
					for b_token, probability in b_tokens.items():
						file.write(f"{label}	{a_token}	{b_token}	{probability}\n")


	def classify(self, text):

		labels = []
		probabilities = []

		text = word_tokenize(Transcriber().transcribe(text.lower(), output="latin"))
		text = [SENTENCE_START] + text + [SENTENCE_END]

		print(" ".join(text))

		for label in self.dataset.keys():	

			labels.append(label)
			
			probability = 0.0
			for index in range(len(text) - 1):
				b_tokens = self.dataset[label].get(text[index], self.dataset[label][UNKNOWN_WORD])
				probability += b_tokens.get(text[index + 1], b_tokens[UNKNOWN_WORD])
			probabilities.append(probability)

		#print(probabilities)

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

	classifier = BigramClassifier()

	classifier.dump()

	classifier.classify("ja videću ga")
	classifier.classify("ja vidjeću ga")
	classifier.classify("ja videt ću ga")
	classifier.classify("ja vidjet ću ga")

	#print(f"Classified as: {result.upper()}")