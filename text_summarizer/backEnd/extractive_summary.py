

import sys
import config
from  Summarizer import *
from spacy.lang.en import STOP_WORDS
import en_core_web_md
from gensim.models import Word2Vec as wv


# Initializing the no of sentences in extracted summary
NUM_OF_SENTENCES = config.NUM_OF_SENTENCES

# Extracting text from input file
print("Step 1: Data Parsing")
parsed_text = pdfparser("learnings.pdf")

final_text = "".join(parsed_text).replace('\n',' ')
text = re.sub('[^A-Za-z0-9%,.$&() ]','',final_text)

print("Step 2: English Language Model loading")
# Loading the Spacy English model
nlp = en_core_web_md.load()


stopwords = {}
for word in STOP_WORDS:
    stopwords[word]=''

print("Step 3: Word Embedding loading")
# Loading the Word Embeddings
model = wv.load('../../../wiki_model_new/wiki.en.text.model')

print("Step 4: Extracting Summary")
summ = Summarizer(model, nlp)
summary = summ.get_summary(text,NUM_OF_SENTENCES)

print ("Summary: ")
print(summary)
