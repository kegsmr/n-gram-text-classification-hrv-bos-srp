import os

class Transcriber:

	phonemes = {}
	latin = {}
	# digraphs = list()

	def __init__(self):
		with open(os.path.join("resources", "alphabet.txt"), encoding="utf-8") as file:
			for line in file:
				cyrillic, latin, ipa = line.strip().split(", ")
				self.phonemes[cyrillic] = ipa
				self.phonemes[latin] = ipa
				self.latin[cyrillic] = latin
		graphemes = sorted(self.phonemes.keys(), key=lambda k: len(k), reverse=True)
		self.phonemes = {grapheme:self.phonemes[grapheme] for grapheme in graphemes}

	def transcribe(self, text, output="ipa"):
		if output == "latin":
			for key in self.latin.keys():
				text = text.replace(key, self.latin[key])
		elif output == "ipa":
			for key in self.phonemes.keys():
				text = text.replace(key, self.phonemes[key])
		return text

if __name__ == "__main__":
    text = "ja čitam knjigu. ја читам књигу"

    transcriber = Transcriber()
    transcription = transcriber.transcribe(text, output="latin")
    print(f"\"{text}\" -> \"{transcription}\"")