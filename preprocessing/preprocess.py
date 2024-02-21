import string
import random
import os
import sys
import math
from nltk.tokenize import word_tokenize

sys.path.append(os.getcwd())

from preprocessing.transcribe import Transcriber

def create_datasets(output_path=os.path.join("resources", "datasets"), corpora=[
			("bos", os.path.join("resources", "corpora", "bos_wikipedia_2021_300K", "bos_wikipedia_2021_300K-sentences.txt")),
			("hrv", os.path.join("resources", "corpora", "hrv_wikipedia_2021_1M", "hrv_wikipedia_2021_1M-sentences.txt")),
			("srp", os.path.join("resources", "corpora", "srp_wikipedia_2021_1M", "srp_wikipedia_2021_1M-sentences.txt")),
			#("eng-au", os.path.join("resources", "corpora", "eng-au_web_2002_300K", "eng-au_web_2002_300K-sentences.txt")),
			#("eng-ca", os.path.join("resources", "corpora", "eng-ca_web_2002_300K", "eng-ca_web_2002_300K-sentences.txt")),
			#("eng-uk", os.path.join("resources", "corpora", "eng-uk_web_2002_300K", "eng-uk_web_2002_300K-sentences.txt")),
		]):

	for label, corpus_path in corpora:

		sentences = []

		print(f"Preprocessing \"{corpus_path}\"...")

		with open(corpus_path, "r", encoding="utf-8") as corpus_file:

			for line in corpus_file:

				sentence = line.split(" ", 1)[1]

				EXCLUDED_CHARACTERS = string.punctuation + string.digits + "“”–"

				for character in EXCLUDED_CHARACTERS:
					sentence = sentence.replace(character, "").lower()

				sentence = Transcriber().transcribe(sentence, output="latin")

				sentences.append(" ".join(word_tokenize(sentence)) + "\n")

		SENTENCES_COUNT = 300000 #len(sentences)

		TRAIN_DEV_TEST_DIRECTORIES = ("train", "dev", "test")
		TRAIN_DEV_TEST_SPLIT = (.6, .2, .2)

		train_dev_test_sentences = (
			[],
			[],
			[],
		)

		for a in range(3):
			for b in range(math.floor(TRAIN_DEV_TEST_SPLIT[a] * SENTENCES_COUNT)):
				train_dev_test_sentences[a].append(sentences.pop(random.randrange(0, len(sentences))))

		for directory in TRAIN_DEV_TEST_DIRECTORIES:
			os.makedirs(os.path.join(output_path, directory), exist_ok=True)

		for i in range(3):
			with open(os.path.join(output_path, TRAIN_DEV_TEST_DIRECTORIES[i], label), "w", encoding="utf-8") as file:
				for sentence in train_dev_test_sentences[i]:
					file.write(sentence)


if __name__ == "__main__":
	create_datasets()