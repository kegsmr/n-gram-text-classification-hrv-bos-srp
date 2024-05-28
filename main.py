from classifiers.unigram import UnigramClassifier
from classifiers.bigram import BigramClassifier
from classifiers.combined import CombinedClassifier


def main():

	unigram_classifier = UnigramClassifier()
	bigram_classifier = BigramClassifier()
	combined_classifier = CombinedClassifier(unigram_classifier=unigram_classifier, bigram_classifier=bigram_classifier)

	classifiers = [("Unigram classifier", unigram_classifier), ("Bigram classifier", bigram_classifier), ("Combined classifier", combined_classifier)]

	while True:

		sentence = input("\n--------------------------------------------\n\nEnter a sentence to classify:\n")
		print()

		for name, classifier in classifiers:
			
			print(f"{name} results:")
			classifier.classify(sentence)


if __name__ == "__main__":
	main()