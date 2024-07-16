# N-Gram Text Classification on Standard Croatian, Bosnian and Serbian

This study attempts to use three different kinds of n-gram text classification models to
differentiate the standard forms of Croatian, Bosnian, and Serbian. These three languages, along
with Montenegrin, were once considered one language, collectively termed “Serbo-Croatian”.
These languages share a common South Slavic ancestry, and there is an argument to be made that
their novel status as distinct languages is due to non-linguistic factors, such as culture and
politics. This study uses 300,000 sentences from each language, sourced from Wikipedia pages
written in the standard forms of each language. Three different classifiers were used: one
unigram, one bigram, and one combined unigram-bigram model. The classifiers were evaluated
by the industry-standard performance metrics for text classification, including accuracy,
precision, recall, and F1. Overall, the classifiers differentiated the three languages relatively
unreliably, with the unigram classifier achieving the highest accuracy rate at 74.05%. The bigram
classifier only achieved a 51.30% accuracy rate, and the combined classifier achieved a 68.94%
accuracy rate. All classifiers were able to predict Serbian sentences most accurately, followed by
Croatian sentences, and Bosnian sentences, with the lowest rate of accuracy. The highest
accuracy overall (87.07%) was achieved using the unigram classifier to predict Serbian
sentences, higher than any other classifier used on any other language. The results suggest that
Croatian, Bosnian and Serbian are too lexically similar to be reliably differentiated on the level
of individual sentences. This may also suggest that the three languages, along with Montenegrin,
should be categorized as one language for practical purposes.
