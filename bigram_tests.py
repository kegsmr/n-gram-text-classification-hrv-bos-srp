from bigram import BigramClassifier
import os

classifier = BigramClassifier()

classifier.dump()

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
ACCURACY
BOS: 25%
HRV: 15%
SRP: 14%
"""

""" 
ACCURACY - on training data ???????????????
BOS: 7%
HRV: 7%
SRP: 7%
"""

""" 
ACCURACY - b_tokens not classified as unknown
BOS: 11%
HRV: 50%
SRP: 48%
"""

""" 
ACCURACY - no single-occurences classified as unknown
BOS: 8%
HRV: 67%
SRP: 66%
"""

""" 
ACCURACY
BOS: 9%
HRV: 74%
SRP: 71%
"""