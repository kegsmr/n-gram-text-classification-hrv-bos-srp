class Transcriber:

    phonemes = dict()
    # digraphs = list()

    def __init__(self):
        with open("alphabet.txt", encoding="utf-8") as file:
            for line in file:
                cyrillic, latin, ipa = line.strip().split(", ")
                self.phonemes[cyrillic] = ipa
                self.phonemes[latin] = ipa
        graphemes = sorted(self.phonemes.keys(), key=lambda k: len(k), reverse=True)
        self.phonemes = {grapheme:self.phonemes[grapheme] for grapheme in graphemes}

    def transcribe(self, text):
        for key in self.phonemes.keys():
            text = text.replace(key, self.phonemes[key])
        return text

if __name__ == "__main__":
    text = "ja čitam knjigu. ја читам књигу"

    transcriber = Transcriber()
    transcription = transcriber.transcribe(text)
    print(f"{text} = {transcription}")