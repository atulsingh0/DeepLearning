import numpy as np
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from collections import Counter as c



def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    return data



class Summarizer(object):

    def __init__(self, model, parser):
        self.model = model
        self.parser = parser


    def tokenizer(self, parsed):
        """
        Tokenizes a sentence into tokens and returns list of words
        and it's mapping to frequencies.

        1. parsed (input): SpaCy parsed text
            * format -> <class 'spacy.tokens.doc.Doc'>

        2. words (output): List of 'unique nouns'
            * format: list

        3. words_to_localcount (output): Mapping of nouns to their frequency
                                         in input text
            * format: dict

        4. words_to_globalcount (output): Mapping of nouns to their frequency
                                          in word2vec model's vocab
            * format: dict

        """

        words=[]
        for sent in parsed.sents:
            for token in sent:
                if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                    words.append(token.text.lower())

        word_to_localcount = c(words)
        words = list(set(words))
        word_to_globalcount = []

        for word in words:
            try:
                glob_count = self.model.wv.vocab[word].count
            except KeyError:
                glob_count = 0

            word_to_globalcount.append((word,glob_count))

        word_to_globalcount = dict(word_to_globalcount)

        print("words, word_to_localcount, word_to_globalcount")
        print(words, word_to_localcount, word_to_globalcount)

        return words, word_to_localcount, word_to_globalcount


    def semantic_centroids(self, nouns, model):
        """
        Finds Semantic Centroids from list of Nouns
        1. nouns (input): List of nouns
            * format -> list
        2. noun_to_score (output): Mapping of Nouns to their relecance weight
            * format: dict
        """

        nouns_score = []

        for noun1 in nouns:
            try:
                self.model[noun1]
                score = 0
                for noun2 in nouns:
                    if noun1!=noun2:
                        try:
                            score+=self.model.similarity(noun1,noun2)
                        except:
                            pass
            except KeyError:
                score = 1
            nouns_score.append((noun1,score))
        noun_to_score = dict(nouns_score)

        print("noun_to_score")
        print(noun_to_score)

        return noun_to_score


    def get_wordweight(self, parsed):
        """
        Tokenizes a sentence into tokens and returns list of words
        and it's mapping to frequencies.

        1. parsed (input): SpaCy parsed text
            * format -> <class 'spacy.tokens.doc.Doc'>

        2. word_to_weight (output): Mapping of Nouns to their respective
                                    Frequency and Semantics based weight
            * format: dict

        """
        words,word_to_localcount,word_to_globalcount = self.tokenizer(parsed)
        centroids = self.semantic_centroids(words, self.model)

        word_to_weight = []

        for word in words:
            weight = np.log(word_to_localcount[word]+1)/np.log((word_to_globalcount[word]+2)**2)*centroids[word]
            word_to_weight.append((word,weight))

        word_to_weight = dict(word_to_weight)
        print("get_wordweight")
        print(parsed, word_to_weight)

        return word_to_weight


    def get_summary(self, text, NUM_OF_SENTENCES):
        """
        Tokenizes a sentence into tokens and returns list of words
        and it's mapping to frequencies.

        1. text (input): Input text to be summarized
            * format -> Unicode

        2. word_to_weight (output): Extractive Summary of the text
            * format: Unicode
        """

        parsed = self.parser(text)
        print ("parsed : ", type(parsed))
        word_to_weight = self.get_wordweight(parsed)

        sents_score=[]

        for sent in parsed.sents:
            if bool(re.search("[0-9]+[ ]+[0-9]", sent.text)) == True or len(sent) < 7:
                sent_score = 0
            else:
                sent_score=0
                for word in sent:
                    try:
                        sent_score+=word_to_weight[word.text.lower()]
                    except KeyError:
                        pass
                sent_score =  sent_score/(len(sent)**(3/4))
            sents_score.append((sent_score,sent.text))

        print("get_summary")
        print(sent_score,sent.text)

        return "\n\n".join([sent for score,sent in sorted(sents_score,reverse=True)[:NUM_OF_SENTENCES]])
