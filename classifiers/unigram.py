import os
from collections import defaultdict
import math
from nltk.tokenize import word_tokenize
import numpy
import sys

sys.path.append(os.getcwd())

from preprocessing.transcribe import Transcriber

class UnigramClassifier:


	def __init__(self, input_path=os.path.join("resources", "datasets", "train"), output_path=os.path.join("resources", "probabilities", "unigram"), labels=["bos", "hrv", "srp"]):

		self.INPUT_PATH = input_path
		self.OUTPUT_PATH = output_path
		self.LABELS = labels

		os.makedirs(self.INPUT_PATH, exist_ok=True)
		os.makedirs(self.OUTPUT_PATH, exist_ok=True)

		self.calculate_probabilities()


	def calculate_probabilities(self, input_paths=None):

		print("Calculating unigram probabilities...")

		if input_paths is None:
			input_paths = [os.path.join(self.INPUT_PATH, dataset) for dataset in self.LABELS]

		self.probabilities = []

		for input_path in input_paths:

			with open(input_path, "r", encoding="utf-8") as input_file:

				type_frequencies = {}
				word_count = 0

				for line in input_file:
					try:
						for token in line.strip().split(" "):
							word_count += 1
							type_frequencies[token] = type_frequencies.get(token, 0) + 1
					except Exception as e:
						print(f"{e}")
				
				type_count = len(type_frequencies)

			print(f"- processed {type_count} types and {word_count} words.")

			default_probability = math.log(1 / (1 + type_count + word_count))

			probability_matrix = defaultdict(lambda: default_probability)

			for type in type_frequencies:

				frequency = type_frequencies[type]

				probability = math.log((1 + frequency) / (1 + type_count + word_count))

				probability_matrix[type] = probability

			self.probabilities.append((os.path.split(input_path)[1], type_count, word_count, probability_matrix))

		self.dump()


	"""def load(self, path=os.path.join("resources", "probabilities", "unigram")):

		self.probabilities = []

		for filename in os.listdir(path):

			with open(os.path.join(path, filename), "r", encoding="utf-8") as file:

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

							probability_matrix = defaultdict(lambda: math.log(1 / (1 + type_count + word_count)))

						else:

							probability_matrix[type] = probability

			self.probabilities.append((filename, type_count, word_count, probability_matrix))


		#total_type_count = sum([item[1] for item in self.probabilities])
		#total_word_count = sum([item[2] for item in self.probabilities])"""

	
	def dump(self):

		for label, type_count, word_count, probability_matrix in self.probabilities:
		
			output_path = os.path.join(self.OUTPUT_PATH, label)

			with open(output_path, "w", encoding="utf-8") as output_file:

				output_file.write(f"{type_count}	{word_count}\n")

				default_probability = math.log(1 / (1 + type_count + word_count))
				output_file.write(f"	{default_probability}\n")

				for type, probability in probability_matrix.items():

					output_file.write(f"{type}	{probability}\n")


	def classify(self, text, only_probabilities=False):

		labels = []
		probabilities = []

		text = word_tokenize(Transcriber().transcribe(text.lower(), output="latin"))
		print(" ".join(text))

		for label, type_count, word_count, probability_matrix in self.probabilities:
			labels.append(label)
			probabilities.append(sum([probability_matrix[token] for token in text])) # + math.log((word_count + type_count) / (total_word_count + total_type_count)))

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

	classifier = UnigramClassifier()

	print()

	classifier.classify("Je postao docentica kemije")
	classifier.classify("Je postao docent hemije")