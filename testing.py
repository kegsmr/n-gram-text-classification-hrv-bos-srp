from classifiers.unigram import UnigramClassifier
from classifiers.bigram import BigramClassifier
from classifiers.combined import CombinedClassifier

unigram_classifier = UnigramClassifier()
bigram_classifier = BigramClassifier()
combined_classifier = CombinedClassifier(unigram_classifier=unigram_classifier, bigram_classifier=bigram_classifier)

classifiers = [unigram_classifier, bigram_classifier, combined_classifier]

while True:

	sentence = input("\n--------------------------------------------\n\nEnter a sentence to classify:\n")
	print()

	for classifier in classifiers:

		classifier.classify(sentence)