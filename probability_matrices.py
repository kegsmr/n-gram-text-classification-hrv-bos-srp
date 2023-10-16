import math
from transcribe import Transcriber

TASKS = [
	("datasets\\train\\bos", "probability-matrices\\bos"),
	("datasets\\train\\hrv", "probability-matrices\\hrv"),
	("datasets\\train\\srp", "probability-matrices\\srp"),
]

transcriber = Transcriber()

for input_path, output_path in TASKS:

	with open(input_path, "r", encoding="utf-8") as input_file:

		type_frequencies = {}
		word_count = 0

		for line in input_file:
			try:
				for token in line.strip().split(" "):
					word_count += 1
					type_frequencies[token] = type_frequencies.get(token, 0) + 1
			except Exception as e:
				print(f"{e}")
		
		type_count = len(type_frequencies)

	print(f"{type_count} types", f"{word_count} words")

	with open(output_path, "w", encoding="utf-8") as output_file:

		output_file.write(f"{type_count}	{word_count}\n")

		default_probability = math.log(1 / (1 + type_count + word_count))
		output_file.write(f"	{default_probability}\n")

		for type in type_frequencies:

			frequency = type_frequencies[type]

			probability = math.log((1 + frequency) / (1 + type_count + word_count))

			output_file.write(f"{type}	{probability}\n")