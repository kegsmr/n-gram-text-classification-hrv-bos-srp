transcriptions = dict()

with open("alphabet.txt", encoding="utf-8") as file:
	for line in file:
		cyrillic, latin, ipa = line.strip().split(", ")
		transcriptions[cyrillic] = ipa
		transcriptions[latin] = ipa

digraphs = [key for key in transcriptions.keys() if len(key) == 2]

text = "ja čitam knjigu. ја читам кнјигу"

def transliterate(text):
	for digraph in digraphs:
		text = text.replace(digraph, transcriptions[digraph])
	for key in transcriptions.keys():
		text = text.replace(key, transcriptions[key])
	return text

transliteration = transliterate(text)
print(transliteration)