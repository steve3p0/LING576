import nltk
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from nltk.corpus import stopwords
from itertools import dropwhile
import os
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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
        self.raw_counts_inserts = self.counter_pos.get('Inserts', 0)

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

    # def passive(self):
    #     """Takes a list of tags, returns true if we think this is a passive
    #     sentence."""
    #     # Particularly, if we see a "BE" verb followed by some other, non-BE
    #     # verb, except for a gerund, we deem the sentence to be passive.
    #
    #     postToBe = list(dropwhile(lambda (tag): not tag.startswith("BE"), tags))
    #     nongerund = lambda (tag): tag.startswith("V") and not tag.startswith("VBG")
    #
    #     filtered = filter(nongerund, postToBe)
    #     out = any(filtered)
    #
    #     return out

# Is Passive function taken from:
# https://github.com/flycrane01/nltk-passive-voice-detector-for-English
def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0:                                                            # no PP, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:                                    # one PP "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']  # gather all the PPs that are not "been".
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                             # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end]
            tagschunk = tags[start:end]
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')] # get all the verbs in between
            if verbspos != []:                                                            # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:  # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
                else:
                    return True
    return False


def main():

    samples = '''I like being hunted.
    The man is being hunted.
    Don't be frightened by what he said.
    I assume that you are not informed of the matter.
    Please be advised that the park is closing soon.
    The book will be released tomorrow.
    We're astonished to see the building torn down.
    The hunter is literally being chased by the tiger.
    He has been awesome since birth.
    She has been beautiful since birth.'''                                                   # "awesome" is wrongly tagged as PP. So the sentence gets a "True".

    sents = nltk.sent_tokenize(samples)
    for sent in sents:
        print(sent + '--> %s' % isPassive(sent))

    #path = 'TedTalks.en-hr.en.txt'
    #path = 'SETIMES.en-hr.en.txt'

    # crp = Corpus('data/bible-uedin.en-hr.en.txt')
    # crp.display_basic_stats()

    # crp = Corpus('data/DGT.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/GNOME.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/hrenWaC.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/KDE4.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/OpenSubtitles.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/QED.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/Tatoeba.en-hr.en.txt')
    # crp.display_basic_stats()

    # crp = Corpus('data/TildeMODEL.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/Ubuntu.en-hr.en.txt')
    # crp.display_basic_stats()
    #
    # crp = Corpus('data/wikimedia.en-hr.en.txt')
    # crp.display_basic_stats()

if __name__ == '__main__':
    # logging.debug("__main__")
    main()