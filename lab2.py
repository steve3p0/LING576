import nltk
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
cause_lemma = lmtzr.lemmatize('cause')
print(cause_lemma)

# cars_lemma = lmtzr.lemmatize('cars')
# feet_lemma = lmtzr.lemmatize('feet')
# pple_lemma = lmtzr.lemmatize('people')
# fant_lemma = lmtzr.lemmatize('fantasized','v')
# print(cars_lemma)

from nltk.corpus import wordnet as wn

def morphify(word,org_pos,target_pos):
    """ morph a word """
    synsets = wn.synsets(word, pos=org_pos)

    # Word not found
    if not synsets:
        return []

    # Get all  lemmas of the word
    lemmas = [l for s in synsets \
                   for l in s.lemmas() if s.name().split('.')[1] == org_pos]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) \
                                    for l in    lemmas]

    # filter only the targeted pos
    related_lemmas = [l for drf in derivationally_related_forms \
                           for l in drf[1] if l.synset().name().split('.')[1] == target_pos]

    # Extract the words from the lemmas
    words = [l.name() for l in related_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result

print (morphify('cause','n','v'))