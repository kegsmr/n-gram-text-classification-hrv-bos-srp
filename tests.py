import os
from unigram import UnigramClassifier
from bigram import BigramClassifier
from combined_unigram_bigram import CombinedUnigramBigramClassifier


DATASETS_PATH = os.path.join("datasets", "test")

LIMIT = None


tests = [
	("unigram", UnigramClassifier(), []),
	("bigram", BigramClassifier(), []),
	("combined", CombinedUnigramBigramClassifier(), []),
]


for name, classifier, results in tests:

	for filename in os.listdir(DATASETS_PATH):

		correct = 0.0
		total = 0.0

		with open(os.path.join(DATASETS_PATH, filename), "r", encoding="utf-8") as file:

			for line in file:

				if LIMIT is not None and total >= LIMIT:
					break

				if classifier.classify(line) == filename:
					correct += 1.0

				total += 1.0
		
		results.append((filename, correct / total))


for name, classsifier, results in tests:

	print(name.upper() + " ACCURACY")

	for type, accuracy in results:
		print(f"{type.upper()}: {int(accuracy * 100)}%")


""" 
UNIGRAM ACCURACY
BOS: 61%
HRV: 71%
SRP: 80%
BIGRAM ACCURACY
BOS: 45%
HRV: 48%
SRP: 54%
COMBINED ACCURACY
BOS: 57%
HRV: 65%
SRP: 73%
"""