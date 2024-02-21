import math
import os


def calculate_unigram_probabilities(tasks=[(
			os.path.join("resources", "datasets", "train", dataset), 
			os.path.join("resources", "probabilities", "unigram", dataset)
		) for dataset in ["bos", "hrv", "srp"]]):

	for input_path, output_path in tasks:

		for path in [input_path, output_path]:
			
			os.makedirs(os.path.split(path)[0], exist_ok=True)

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

	
def calculate_bigram_probabilities(tasks=[(
			os.path.join("resources", "datasets", "train", dataset), 
			os.path.join("resources", "probabilities", "bigram", dataset)
		) for dataset in ["bos", "hrv", "srp"]]):
	
	for input_path, output_path in tasks:

		for path in [input_path, output_path]:
			
			os.makedirs(os.path.split(path)[0], exist_ok=True)

	


if __name__ == "__main__":
	calculate_unigram_probabilities()
	calculate_bigram_probabilities()