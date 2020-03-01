import nltk
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
path = 'C:\workspace_courses\LING576\corpora\LSWE Corpus\BrE Conv'
files = os.listdir(path)

raw = ""

for file in files:
    if os.path.isfile(os.path.join(path, file)):
        f = open(os.path.join(path, file), 'r')
        raw += re.sub('<[^<]+?>', '', f.read())
        # print(f"raw: {raw}")
        f.close()

tokens = nltk.word_tokenize(raw)
text = nltk.Text(tokens)

print(f"raw: {raw[:50]}")
print(f"tokens: {tokens}")
print(f"text: {text[:50]}")

token_count = len(tokens)
file_count = len(files)

print(f"file count: {file_count}")
print(f"word count: {token_count}")

# KWIC of 'stuff'
word = 'stuff'
kwic_word_count = text.count(word)
print(f"\nKWIC count of '{word}': {kwic_word_count}")
print(f"KWIC of '{word}':\n {text.concordance(word)}")

# KWIC of 'stuff like that'
phrase = 'stuff like that'
kwic_count = text.count(phrase)
print(f"\nKWIC count of '{phrase}: {kwic_word_count}")
print(f"KWIC of '{phrase}':\n {text.concordance(phrase)}")
