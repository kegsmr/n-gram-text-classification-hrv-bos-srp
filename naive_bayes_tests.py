import naive_bayes
import os

DATASETS_PATH = os.path.join("datasets", "dev")

LIMIT = 100

results = []

for filename in os.listdir(DATASETS_PATH):

	correct = 0.0
	total = 0.0

	with open(os.path.join(DATASETS_PATH, filename), "r", encoding="utf-8") as file:

		for line in file:

			if LIMIT is not None and total >= LIMIT:
				break

			if naive_bayes.classify(line) == filename:
				correct += 1.0

			total += 1.0
	
	results.append((filename, correct / total))

print('\nACCURACY')

for type, accuracy in results:
	print(f"{type.upper()}: {int(accuracy * 100)}%")

""" (with class probability)
ACCURACY 
BOS: 37%
HRV: 84%
SRP: 88%
"""

""" (without class probability)
ACCURACY
BOS: 60%
HRV: 73%
SRP: 82%
"""