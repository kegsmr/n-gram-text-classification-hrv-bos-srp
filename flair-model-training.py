from flair.datasets import UD_ENGLISH
from flair.embeddings import WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. load the corpus
corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

# 2. what label do we want to predict?
label_type = 'upos'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embeddings = WordEmbeddings('glove')

# 5. initialize sequence tagger
model = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(model, corpus)

# 7. start training
trainer.train('resources/taggers/example-upos',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)