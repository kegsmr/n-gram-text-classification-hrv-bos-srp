from naive_bayes import NaiveBayesClassifier
import os

classifier = NaiveBayesClassifier()

DATASETS_PATH = os.path.join("datasets", "dev")

LIMIT = None

results = []

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

print('ACCURACY')

for type, accuracy in results:
	print(f"{type.upper()}: {int(accuracy * 100)}%")

""" 
ACCURACY (100 per type, with class probability)
BOS: 37%
HRV: 84%
SRP: 88%
"""

""" 
ACCURACY (100 per type, without class probability)
BOS: 60%
HRV: 73%
SRP: 82%
"""

"""
ACCURACY (All)
BOS: 61%
HRV: 75%
SRP: 81%
"""