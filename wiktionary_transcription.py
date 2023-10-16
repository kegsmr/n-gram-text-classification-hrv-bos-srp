import wiktionaryparser
import nltk

nltk.download('punkt')

text = "Radovan Karadžić se rodio u selu Petnjica, općina Šavnik u Crnoj Gori. Prema nekim izvorima na istom je mjestu rođen i otac Vuka Karadžića. Završio je osnovnu školu u Nikšiću, a potom se s 15 godina preselio u Sarajevo gdje je završio srednju medicinsku školu, a potom studije medicine, da bi na kraju specijalizirao psihijatriju (što je i doktorirao). Dio školovanja je obavio u SAD-u, a u sarajevskoj bolnici na Koševu je pacijente liječio od depresije. Osim medicinom, Karadžić se bavio i pisanjem poezije. Oženjen je s Ljiljanom Zelen-Karadžić, s kojom ima dvoje djece - kći Sonju i sina Sašu. Godine 1987. je završio u pritvoru pod optužbom da je pronevjerio novac kako bi sagradio vikendicu na Palama. U pritvoru je proveo 11 mjeseci, ali je poslije pušten."

tokens = nltk.word_tokenize(text)

for token in tokens:
	try:
		print(wiktionaryparser.WiktionaryParser().fetch(word, "serbo-croatian")[0]["pronunciations"]["text"][0].split("/")[1])
	except:
		print("\"" + token + "\" NOT AVAILABLE")