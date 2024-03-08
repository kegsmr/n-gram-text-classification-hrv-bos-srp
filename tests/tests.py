import os
import sys
import csv

sys.path.append(os.getcwd())

from classifiers.unigram import UnigramClassifier
from classifiers.bigram import BigramClassifier
from classifiers.combined import CombinedClassifier
from preprocessing.preprocess import create_datasets


DATASETS_PATH = os.path.join("resources", "datasets", "test")

LIMIT = None


unigram_classifier = UnigramClassifier()
bigram_classifier = BigramClassifier()

tests = [
	("unigram", unigram_classifier, []),
	("bigram", bigram_classifier, []),
	("combined", CombinedClassifier(unigram_classifier=unigram_classifier, bigram_classifier=bigram_classifier), []),
]


"""performance = {}

for name in [test[0] for test in tests]:
	performance[name] = {}
	for label in os.listdir(DATASETS_PATH):
		performance[name][label] = {}
		for category in ["TP", "TN", "FP", "FN"]:
			performance[name][label][category] = 0"""


with open("predictions.csv", "w", encoding="utf-8", newline='') as csv_file:

	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(["Classifier", "Actual", "Predicted", "Sentence"])

	for name, classifier, results in tests:

		#overall_correct = 0.0
		#overall_total = 0.0

		for label in os.listdir(DATASETS_PATH):

			correct = 0.0
			total = 0.0

			with open(os.path.join(DATASETS_PATH, label), "r", encoding="utf-8") as file:

				for line in file:

					if LIMIT is not None and total >= LIMIT:
						break

					predicted_label = classifier.classify(line)

					csv_writer.writerow([name, label, predicted_label, line])

					if label == predicted_label:
						#performance[name][label][""]
						correct += 1.0

					total += 1.0
			
			#overall_correct += correct
			#overall_total += total

			results.append((label, correct / total))


for name, classsifier, results in tests:

	print(name.upper() + " ACCURACY:")

	for type, accuracy in results:
		print(f"{type.upper()}: {int(accuracy * 100)}%")


#print(f"OVERALL ACCURACY: {int((overall_correct / overall_total) * 100)}%")


""" 
UNIGRAM ACCURACY
BOS: 66%
HRV: 73%
SRP: 81%
BIGRAM ACCURACY
BOS: 45%
HRV: 49%
SRP: 56%
COMBINED ACCURACY
BOS: 60%
HRV: 66%
SRP: 75%
"""

"""
UNIGRAM ACCURACY
BOS: 62%
HRV: 72%
SRP: 80%
BIGRAM ACCURACY
BOS: 41%
HRV: 55%
SRP: 50%
COMBINED ACCURACY
BOS: 56%
HRV: 70%
SRP: 72%
"""

"""
UNIGRAM ACCURACY:
BOS: 62%
HRV: 73%
SRP: 80%
BIGRAM ACCURACY:
BOS: 43%
HRV: 46%
SRP: 59%
COMBINED ACCURACY:
BOS: 58%
HRV: 65%
SRP: 77%
"""

""" 
UNIGRAM ACCURACY:
BOS: 65%
HRV: 74%
SRP: 82%
BIGRAM ACCURACY:
BOS: 44%
HRV: 49%
SRP: 60%
COMBINED ACCURACY:
BOS: 59%
HRV: 68%
SRP: 78%
"""