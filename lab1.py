import nltk
from nltk.collocations import *
import os
import re
import pandas as pd

# filename = 'AP900124.TXT'
# file = open(filename,'r')
# raw = file.read()
# tokens = nltk.word_tokenize(raw)
# text = nltk.Text(tokens)
#
# print(f"raw: {raw}")
# print(f"tokens: {tokens}")
# print(f"text: {text}")
#
# print(f"\n\n")

# TODO: What to do with headers?
# <DOC>
# <DOCNO> AP900124-0001 </DOCNO>
# <FILEID>AP-NR-01-24-90 0135EST</FILEID>
# <FIRST>r a AM-DateRape     01-24 0400</FIRST>
# <SECOND>AM-Date Rape,0414</SECOND>
# <HEAD>Man Gets Long Prison Sentence For ``Date Rape'' Incidents</HEAD>
# <DATELINE>ALBANY, N.Y. (AP) </DATELINE>
# <TEXT>


#######################################
def load_corpus(path, encoding='utf-8'):
    files = os.listdir(path)
    raw = ""
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            f = open(os.path.join(path, file), 'r', encoding=encoding)
            raw += re.sub('<[^<]+?>', '', f.read())
            # print(f"raw: {raw}")
            f.close()

    return raw, len(files)

def n_concordance_tokenised(text,phrase,left_margin=5,right_margin=5):
    #concordance replication via https://simplypython.wordpress.com/2014/03/14/saving-output-of-nltk-text-concordance/

    phraseList=phrase.split(' ')

    c = nltk.ConcordanceIndex(text.tokens, key = lambda s: s.lower())

    #Find the offset for each token in the phrase
    offsets=[c.offsets(x) for x in phraseList]
    offsets_norm=[]
    #For each token in the phraselist, find the offsets and rebase them to the start of the phrase
    for i in range(len(phraseList)):
        offsets_norm.append([x-i for x in offsets[i]])
    #We have found the offset of a phrase if the rebased values intersect
    #--
    # http://stackoverflow.com/a/3852792/454773
    #the intersection method takes an arbitrary amount of arguments
    #result = set(d[0]).intersection(*d[1:])
    #--
    intersects=set(offsets_norm[0]).intersection(*offsets_norm[1:])

    concordance_txt = ([text.tokens[map(lambda x: x-left_margin if (x-left_margin) > 0 else 0,[offset])[0]:offset+len(phraseList)+right_margin]
                    for offset in intersects])

    outputs=[''.join([x+' ' for x in con_sub]) for con_sub in concordance_txt]
    return outputs

def n_concordance(txt, phrase, left_margin=5, right_margin=5):
    tokens = nltk.word_tokenize(txt)
    text = nltk.Text(tokens)

    return

# n_concordance_tokenised(text, phrase, left_margin=left_margin, right_margin=right_margin)
# n_concordance_tokenised(text1, 'monstrous size')


#path = 'corpora\LSWE Corpus\BrE Conv'
path = 'corpora\LSWE Corpus\AmE News'
raw, file_count = load_corpus(path)
tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)

print(f"raw: {raw[:50]}")
print(f"tokens: {tokens}")
print(f"text: {text[:50]}")

token_count = len(tokens)

print(f"file count: {file_count}")
print(f"word count: {token_count}")


# 4. Run a KWIC of stuff in the newspapers files
word = 'stuff'
kwic_word_count = text.count(word)
print(f"\nKWIC count of '{word}': {kwic_word_count}")
print(f"KWIC of '{word}':\n {text.concordance(word)}")

# KWIC of 'stuff like that'
phrase = 'stuff like that'
kwic_count = text.count(phrase)
print(f"\nKWIC count of '{phrase}: {kwic_word_count}")
print(f"KWIC of '{phrase}':\n {text.concordance(phrase)}")

# c) How does the frequency of stuff in newspapers compare to the frequency in conversation?
#    Does the comparison suprise you?  Why/why not?

path = 'corpora\LSWE Corpus\AmE Conv'
raw, file_count = load_corpus(path, encoding='latin-1')
word = 'stuff'
kwic_word_count = text.count(word)
print(f"\nKWIC count of '{word}': {kwic_word_count}")
print(f"KWIC of '{word}':\n {text.concordance(word)}")

# 5. Choose your own word to run a concordance of in the newspaper files.
path = 'corpora\LSWE Corpus\AmE News'
raw, file_count = load_corpus(path)
word = 'mobile'
print(f"\na) Word: {word}")
kwic_word_count = text.count(word)
print(f"\nb) Number of occurances: {kwic_word_count}")
# 	c)  Sort by right or left (whichever is most interesting to you).  Are there any sequences
# 	     that are particularly common as you eye the KWIC file?
print(f" c)  Sort by right or left (whichever is most interesting to you).")
print(f"     Are there any sequences that are particularly common as you eye the KWIC file?")
print(f" NOTE: There is no built-in sort functionality for the concordance function in the NLTK.")
print(f"KWIC of '{word}':\n {text.concordance(word)}")

# 6. Compare the use of threw in American newspapers and American conversation.
# 	a) raw counts:  News________________     Conv___________________
news_path = 'corpora\LSWE Corpus\AmE News'
news_raw, news_file_count = load_corpus(news_path)
word = 'threw'
news_word_count = text.count(word)

#from nltk.app import concordance
phrase = 'threw for'
#news_conc = concordance()
news_tokens = nltk.word_tokenize(news_raw)
news_text = nltk.Text(news_tokens)
# news_conc = n_concordance_tokenised(news_text, phrase)

#news_conc = text.concordance_list(phrase)
# from  nltk.text import ConcordanceIndex
# ci = ConcordanceIndex(text.tokens)
# results = concordance(ci, 'circumstances')
#news_conc = text.concordance_list(phrase)

conv_path = 'corpora\LSWE Corpus\AmE Conv'
raw, file_count = load_corpus(conv_path, encoding='latin-1')
word = 'threw'
conv_word_count = text.count(word)
print(f"\na) raw counts -  News: {news_word_count}; Conversation: {conv_word_count}")
# 	b) Look at the occurrences of threw for in the newspapers.  In what context is this
# 	      sequence used most often?  (Be as specific as you can about the context.)
print(f"b) Look at the occurrences of threw for in the newspapers.")
print(f"   In what context is this sequence used most often?")
print(f"    (Be as specific as you can about the context.)")
# print(f"   Concordance:\n {news_conc}")

# 	c) How often is threw for used in the conversations?
# Issue with multi-word concordance searching


#################################33
# Part B. Investigatinv Collacates

# 1.	Before you start: If someone asked you quickly to tell the meaning of the word "great," what would you say?

news_path = 'corpora\LSWE Corpus\AmE News'
news_raw, news_file_count = load_corpus(news_path)
news_tokens = nltk.word_tokenize(news_raw)
news_text = nltk.Text(news_tokens)

word = 'great'
news_word_count = text.count(word)

# 2. Load all the American newspaper files and run a concordance of great. How many occurrences are there?
print(f"2. Load all the American newspaper files and run a concordance of great.")
print(f"   How man occurrences are there? {news_word_count}")

# 3. What are the 3 most frequent immediate left collocates of great?
bigram_measures = nltk.collocations.BigramAssocMeasures()

# Ngrams with 'great' as a member
word = 'great'
great_filter = lambda *w: word not in w

## Bigrams
finder = BigramCollocationFinder.from_words(news_text)
finder.apply_freq_filter(3)
finder.apply_ngram_filter(great_filter)
d = dict(finder.ngram_fd)
left_list = [(k[0], v) for k,v in d.items() if k[1] == 'great']
right_list = [(k[1], v) for k,v in d.items() if k[0] == 'great']

left_list.sort(key = lambda x: x[1])
right_list.sort(key = lambda x: x[1])
print(f"What are the 3 most frequent immediate left collocates of great? \n{left_list[:3]}")
print(f"And the 3 most frequent immediate right collocates of great? \n{right_list[:3]}")

# 4. Look at the patterns for a series of 3 words with great in the middle (i.e., first left – great – first right).
#    What is the most common sequence that does not use a proper noun?
## Trigrams
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(news_text)
finder.apply_freq_filter(3)
finder.apply_ngram_filter(great_filter)
d = dict(finder.ngram_fd)
middle_list = [(k[0], v) for k,v in d.items() if k[1] == 'great']
middle_list.sort(key = lambda x: x[1])
print(f"What is the most common sequence that does not use a proper noun?")
print(f"NOTE: eyeball the top 10")
print(f"{middle_list[:3]}")


#
# 5.	Run a concordance listing with the 3-word sequence you found in #4.
# 	What is the most common first right collocate of the 3-word sequence?
#
#
# 6. 	Now, run a concordance listing with the 4-word sequence you found in #5.
# 	A)  What is the most common first right collocate that is a noun?
#
#
# 	B)  How many occurrences of the most common noun are there?
#
#
# 7. 	How do your findings here compare with your quick definition of "great" in #1?
