import math
from transcribe import Transcriber

TASKS = [
	("wikipedia-corpora\\bos_wikipedia_2021_300K\\bos_wikipedia_2021_300K-words.txt", "probability-matrices\\bos"),
	("wikipedia-corpora\\hrv_wikipedia_2021_1M\\hrv_wikipedia_2021_1M-words.txt", "probability-matrices\\hrv"),
	("wikipedia-corpora\\srp_wikipedia_2021_1M\\srp_wikipedia_2021_1M-words.txt", "probability-matrices\\srp"),
]

transcriber = Transcriber()

for input_path, output_path in TASKS:

	with open(input_path, "r", encoding="utf-8") as input_file:

		type_frequencies = []

		word_count = 0
		type_count = 0

		for line in input_file:

			try:
			
				number, type, frequency = line.split("	")

				type = transcriber.transcribe(type, output="latin")
				frequency = int(frequency)

				word_count += frequency
				type_count += 1

				type_frequencies.append((type, frequency))

			except Exception as e:

				print(f"{e}")

	print(f"{type_count} types", f"{word_count} words")

	with open(output_path, "w", encoding="utf-8") as output_file:

		default_probability = math.log(1 / (1 + type_count + word_count))

		output_file.write(f"	{default_probability}\n")

		for type, frequency in type_frequencies:

			probability = math.log((1 + frequency) / (1 + type_count + word_count))

			output_file.write(f"{type}	{probability}\n")