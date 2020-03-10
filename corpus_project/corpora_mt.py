import nltk
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from nltk.corpus import stopwords
import os
import re

#######################################

class Corpus:
    def __init__(self, path, encoding='utf-8'):

        self.path = path
        if os.path.isdir(path):
            self.files = os.listdir(path)
        else:
            self.files = path

        self.raw = ""
        for file in self.files:
            if os.path.isfile(os.path.join(path, file)):
                f = open(os.path.join(path, file), 'r', encoding=encoding)
                self.raw += re.sub('<[^<]+?>', '', f.read())
                # print(f"raw: {raw}")
                f.close()

        self.tokens = nltk.word_tokenize(self.raw)
        self.sentences = nltk.sent_tokenize(self.raw)
        self.text = nltk.Text(self.tokens)
        stop_words = set(stopwords.words('english'))
        self.text_no_stop = nltk.Text([w.lower() for w in self.tokens if w.lower() not in stop_words])

        # Get syntax props
        self._get_syntax()

        # Calculate Totals
        self._get_counts()

    def _get_syntax(self):
        # Tagging Style 1
        self.words = defaultdict(list)
        insert_words = ('yeah', 'Ok', 'ahh')

        self.tags1 = nltk.pos_tag(self.text)
        self.tags2 = nltk.pos_tag(self.text, tagset='universal')
        # Build a dictionary of words, grouped by POS
        for w, tag in self.tags2:
            if w in insert_words:
                tag = "Inserts"
                self.words[tag].append(w)
            elif tag in ('PRON', 'DET', 'ADP', 'CONJ'):
                tag = "FUNCTOR"
                self.words[tag].append(w)
            elif tag != '.':
                self.words[tag].append(w)
            else:
                self.words[tag].append(w)

        self.counter_pos = {k: len(v) for k, v in self.words.items()}

    def _get_counts(self):

        self.total_files = len(self.files)
        self.total_sentences = len(self.sentences)
        self.total_words = sum(self.counter_pos.values())
        self.token_count = len(self.tokens)  # Total number of tokens
        self.token_unique = len(set(self.tokens))  # number of unique tokens
        self.lexical_diversity = self.token_unique / self.token_count  # lexical diversity
        self.average_sentence_length = self.token_count / self.total_sentences

        self.freq_dist = nltk.FreqDist(self.text_no_stop)

        self.raw_counts_nouns = self.counter_pos['NOUN']
        self.raw_counts_verbs = self.counter_pos['VERB']
        self.raw_counts_adverbs = self.counter_pos['ADV']
        self.raw_counts_adjectives = self.counter_pos['ADJ']
        self.raw_counts_functors = self.counter_pos['FUNCTOR']
        self.raw_counts_inserts = self.counter_pos['Inserts']

        self.percent_nouns = self.raw_counts_nouns / self.total_words
        self.percent_verbs = self.raw_counts_verbs / self.total_words
        self.percent_adjectives = self.raw_counts_adjectives / self.total_words
        self.percent_adverbs = self.raw_counts_adverbs / self.total_words
        self.percent_functors = self.raw_counts_functors / self.total_words
        self.percent_inserts = self.raw_counts_inserts / self.total_words

        self.basis = 1000

        self.norm_counts_nouns = self.percent_nouns * self.basis
        self.norm_counts_verbs = self.percent_verbs * self.basis
        self.norm_counts_adjectives = self.percent_adjectives * self.basis
        self.norm_counts_adverbs = self.percent_adverbs * self.basis
        self.norm_counts_functors = self.percent_functors * self.basis
        self.norm_counts_inserts = self.percent_inserts * self.basis


    def display_basic_stats(self):
        print(f"\n\nBasic NLP Statistics for Corpus '{self.path}':")
        print(f"\tFile Count: {self.total_files}")
        print(f"\tToken Count: {self.token_count}")
        print(f"\tUnique Count: {self.token_unique}")
        print(f"\tSentence Count: {self.total_sentences}")
        print(f"\tAverage Sentence Length: {self.average_sentence_length}")
        print(f"\tLexical Diversity: {self.lexical_diversity}")
        #print(f"\n\nTAGS:\n{self.tags1}")
        #print(f"\n\nTAGS:\n{self.tags2}")

        # Print Counts
        print(f"\n\nRaw Counts:")
        print(f"\tNouns: {self.raw_counts_nouns}")
        print(f"\tVerbs: {self.raw_counts_verbs}")
        print(f"\tAdjectives: {self.raw_counts_adjectives}")
        print(f"\tAdverbs: {self.raw_counts_adverbs}")
        print(f"\tFunction Words: {self.raw_counts_functors}")
        print(f"\tInserts: {self.raw_counts_inserts}")
        print(f"\tTotal Words: {self.total_words}")

        print(f"\n\nPercentages:")
        print(f"\tNouns: {self.percent_nouns:.1%}")
        print(f"\tVerbs: {self.percent_verbs:.1%}")
        print(f"\tAdjectives: {self.percent_adjectives:.1%}")
        print(f"\tAdverbs: {self.percent_adverbs:.1%}")
        print(f"\tFunction Words: {self.percent_functors:.1%}")
        print(f"\tInserts: {self.percent_inserts:.1%}")
        total_percentages = sum([self.percent_nouns, self.percent_verbs, self.percent_adjectives, self.percent_adverbs,
                                 self.percent_functors, self.percent_inserts])
        print(f"\tTotal Percentages: {total_percentages:.1%}")

        print(f"\n\nNormed Counts Per {self.basis}:")
        print(f"\tNouns: {self.norm_counts_nouns:0.1f}")
        print(f"\tVerbs: {self.norm_counts_verbs:0.1f}")
        print(f"\tAdjectives: {self.norm_counts_adjectives:0.1f}")
        print(f"\tAdverbs: {self.norm_counts_adverbs:0.1f}")
        print(f"\tFunction Words: {self.norm_counts_functors:0.1f}")
        print(f"\tInserts: {self.norm_counts_inserts:0.1f}")
        total_norm_counts = sum([self.norm_counts_nouns, self.norm_counts_verbs, self.norm_counts_adjectives, self.norm_counts_adverbs,
                                 self.norm_counts_functors, self.norm_counts_inserts])
        print(f"\tTotal Norm Counts: {total_norm_counts:0.1f}")

def main():

    #path = 'TedTalks.en-hr.en.txt'
    path = 'SETIMES.en-hr.en.txt'
    crp = Corpus(path)
    crp.display_basic_stats()


    # raw, file_count = load_corpus(path)
    # tokens = nltk.word_tokenize(raw)
    # text = nltk.Text(tokens)
    # token_count = len(tokens)
    #
    # print(f"raw: {raw[:50]}")
    # print(f"tokens: {tokens}")
    # print(f"text[1st 50 sentences]: {text[:50]}")
    # print(f"file count: {file_count}")
    # print(f"word count: {token_count}")
    #
    # # 4. Run a KWIC of stuff in the newspapers files
    # word = 'stuff'
    # kwic_word_count = text.count(word)
    # print(f"\nKWIC count of '{word}': {kwic_word_count}")
    # print(f"KWIC of '{word}':\n {text.concordance(word)}")
    #
    # # KWIC of 'stuff like that'
    # phrase = 'stuff like that'
    # kwic_count = text.count(phrase)
    # print(f"\nKWIC count of '{phrase}: {kwic_word_count}")
    # print(f"KWIC of '{phrase}':\n {text.concordance(phrase)}")
    #
    # ##############################
    # # Tagging
    # tagged_text = nltk.pos_tag(text)
    #
    # print(f"tagged text: {tagged_text}")


if __name__ == '__main__':
    # logging.debug("__main__")
    main()