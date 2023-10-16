import nltk
import string
import random
import os
import math
from nltk.tokenize import word_tokenize
from transcribe import Transcriber

transcriber = Transcriber()

CORPORA = [
	("bos", "wikipedia-corpora\\bos_wikipedia_2021_300K\\bos_wikipedia_2021_300K-sentences.txt"),
	("hrv", "wikipedia-corpora\\hrv_wikipedia_2021_1M\\hrv_wikipedia_2021_1M-sentences.txt"),
	("srp", "wikipedia-corpora\\srp_wikipedia_2021_1M\\srp_wikipedia_2021_1M-sentences.txt"),
]

for label, corpus_path in CORPORA:

	sentences = []

	with open(corpus_path, "r", encoding="utf-8") as corpus_file:

		for line in corpus_file:

			sentence = line.split(" ", 1)[1]

			EXCLUDED_CHARACTERS = string.punctuation + string.digits + "“”"

			for character in EXCLUDED_CHARACTERS:
				sentence = sentence.replace(character, "").lower()

			sentence = transcriber.transcribe(sentence, output="latin")

			sentences.append(" ".join(word_tokenize(sentence)) + "\n")

	sentences_count = len(sentences)

	TRAIN_DEV_TEST_DIRECTORIES = ("train", "dev", "test")
	TRAIN_DEV_TEST_SPLIT = (.6, .2, .2)

	train_dev_test_sentences = (
		[],
		[],
		[],
	)

	for a in range(3):
		for b in range(math.floor(TRAIN_DEV_TEST_SPLIT[a] * sentences_count)):
			train_dev_test_sentences[a].append(sentences.pop(random.randrange(0, len(sentences))))

	for i in range(3):
		with open(os.path.join("datasets", TRAIN_DEV_TEST_DIRECTORIES[i], label), "w", encoding="utf-8") as file:
			for sentence in train_dev_test_sentences[i]:
				file.write(sentence)