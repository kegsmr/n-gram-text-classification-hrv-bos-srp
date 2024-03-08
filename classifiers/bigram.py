import sys
import math
import os
from nltk.tokenize import word_tokenize
import numpy

sys.path.append(os.getcwd())

from preprocessing.transcribe import Transcriber

class BigramClassifier:

	
	def __init__(self, input_path = os.path.join("resources", "datasets", "train"), output_path = os.path.join("resources", "probabilities", "bigram"), labels=["bos", "hrv", "srp"]):

		self.INPUT_PATH = input_path
		self.OUTPUT_PATH = output_path
		self.LABELS = labels

		self.SENTENCE_START = "[S]"
		self.SENTENCE_END = "[E]"
		self.UNKNOWN_WORD = "<UNK>"

		os.makedirs(self.INPUT_PATH, exist_ok=True)
		os.makedirs(self.OUTPUT_PATH, exist_ok=True)

		self.calculate_probabilities()


	def calculate_probabilities(self, input_paths=None):

		print("Calculating bigram probabilities...")

		if input_paths is None:
			input_paths = [os.path.join(self.INPUT_PATH, dataset) for dataset in self.LABELS]

		b = {}

		for input_path in input_paths:

			label = os.path.split(input_path)[1]

			b.setdefault(label, {})
			b[label].setdefault(self.UNKNOWN_WORD, {})
			b[label][self.UNKNOWN_WORD].setdefault(self.UNKNOWN_WORD, 0.0)

			with open(input_path, "r", encoding="utf-8") as file:

				for line in file:
					
					try:

						tokens = line.strip().split(" ")
						tokens = [self.SENTENCE_START] + tokens + [self.SENTENCE_END]
						
						for index in range(len(tokens) - 1):
							
							a_token = tokens[index]
							b_token = tokens[index + 1]
							
							b[label].setdefault(a_token, {})
							b[label][a_token].setdefault(b_token, 0.0)
							b[label][a_token][b_token] += 1.0

							b[label][a_token].setdefault(self.UNKNOWN_WORD, 0.0)

					except Exception as e:

						print(f"{e}")

		self.bigrams = b

		b = {}

		for label, a_tokens in self.bigrams.items():
			b.setdefault(label, {})
			for a_token, b_tokens in a_tokens.items():
				b[label].setdefault(a_token, {})
				for b_token, frequency in b_tokens.items():
					b[label][a_token][b_token] = math.log((1.0 + frequency) / (1.0 + len(b[label][a_token].values()) + sum(b[label][a_token].values())))

		self.bigrams = b

		self.dump()


	def dump(self):
		for label, a_tokens in self.bigrams.items():
			with open(os.path.join(self.OUTPUT_PATH, label), "w", encoding="utf-8") as file:
				for a_token, b_tokens in a_tokens.items():
					for b_token, probability in b_tokens.items():
						file.write(f"{a_token}	{b_token}	{probability}\n")


	def classify(self, text, only_probabilities=False):

		labels = []
		probabilities = []

		text = word_tokenize(Transcriber().transcribe(text.lower(), output="latin"))
		text = [self.SENTENCE_START] + text + [self.SENTENCE_END]

		print(" ".join(text))

		for label in self.bigrams.keys():	

			labels.append(label)
			
			probability = 0.0
			for index in range(len(text) - 1):
				b_tokens = self.bigrams[label].get(text[index], self.bigrams[label][self.UNKNOWN_WORD])
				new_probability = b_tokens.get(text[index + 1], b_tokens[self.UNKNOWN_WORD])
				print(f"{label} {text[index]} {text[index + 1]} {new_probability}")
				probability += new_probability
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

	classifier = BigramClassifier()

	classifier.classify("ja videću ga")
	classifier.classify("ja vidjeću ga")
	classifier.classify("ja videt ću ga")
	classifier.classify("ja vidjet ću ga")