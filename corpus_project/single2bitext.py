import nltk

# Reference:
# This code comes from here:
# https://github.com/flycrane01/nltk-passive-voice-detector-for-English
def isPassive(sentence):

    # all forms of "be"
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']

    # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    aux = ['do', 'did', 'does', 'have', 'has', 'had']
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]

    # no PP, no passive voice.
    if tags.count('VBN') == 0:
        return False
    # one PP "been", still no passive voice.
    elif tags.count('VBN') == 1 and 'been' in words:
        return False
    else:
        # gather all the PPs that are not "been".
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']

        for end in pos:
            chunk = tags[:end]
            start = 0

            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                if last == 'NN' or last == 'PRP':
                    start = i
                    break

            sentchunk = words[start:end]
            tagschunk = tags[start:end]

            # get all the verbs in between
            verbspos = [i for i in range(len(tagschunk)) if  tagschunk[i].startswith('V')]

            # if there are no verbs in between, it's not passive
            if verbspos != []:
                for i in verbspos:
                    # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:
                        break
                else:
                    return True

    return False

def single2bitext(src, tgt, merge_p, merge_np):
    with open(merge_p, 'w', encoding="utf8") as passive, \
         open(merge_np, 'w', encoding="utf8") as not_passive, \
         open(src, encoding="utf8") as f1, \
         open(tgt, encoding="utf8") as f2:
        for line1, line2 in zip(f1, f2):
            if isPassive(line2):
                passive.write("{0}\t{1}\n".format(line1.rstrip(), line2.rstrip()))
            else:
                not_passive.write("{0}\t{1}\n".format(line1.rstrip(), line2.rstrip()))


if __name__ == '__main__':
    # source = 'enhr/prep/paracrawl.test10k.hr'
    # target = 'enhr/prep/paracrawl.test10k.en'
    # merge = "enhr/prep/paracrawl.test10k.hr-en.txt"

    # source = 'enhr/prep/paracrawl.tune10k.hr'
    # target = 'enhr/prep/paracrawl.tune10k.en'
    # merge = "enhr/prep/paracrawl.tune10k.hr-en.txt"

    # source = 'enhr/prep/paracrawl.train.hr'
    # target = 'enhr/prep/paracrawl.train.en'
    # merge = "enhr/prep/paracrawl.train.hr-en.txt"

    # source = 'enhr/prep/ted2013.tune.hr'
    # target = 'enhr/prep/ted2013.tune.en'
    # merge = "enhr/prep/ted2013.tune.hr-en.txt"

    ###################################################333

    # source = 'enhr/prep/paracrawl.test10k.hr'
    # target = 'enhr/prep/paracrawl.test10k.en'
    # merge_passive = "enhr/prep/paracrawl.test10k.passive.hr-en.txt"
    # merge_not_passive = "enhr/prep/paracrawl.test10k.not-passive.hr-en.txt"

    # source = 'enhr/prep/paracrawl.tune10k.hr'
    # target = 'enhr/prep/paracrawl.tune10k.en'
    # merge_passive = "enhr/prep/paracrawl.tune10k.passive.hr-en.txt"
    # merge_not_passive = "enhr/prep/paracrawl.tune10k.not-passive.hr-en.txt"

    # source = 'enhr/prep/ted2013.test.hr'
    # target = 'enhr/prep/ted2013.test.en'
    # merge_passive = "enhr/prep/ted2013.test.passive.hr-en.txt"
    # merge_not_passive = "enhr/prep/ted2013.test.not-passive.hr-en.txt"

    # source = 'enhr/prep/ted2013.tune.hr'
    # target = 'enhr/prep/ted2013.tune.en'
    # merge_passive = "enhr/prep/ted2013.tune.passive.hr-en.txt"
    # merge_not_passive = "enhr/prep/ted2013.tune.not-passive.hr-en.txt"

    source = 'enhr/prep/paracrawl.train.hr'
    target = 'enhr/prep/paracrawl.train.en'
    merge_passive = "enhr/prep/paracrawl.train.passive.hr-en.txt"
    merge_not_passive = "enhr/prep/paracrawl.train.not-passive.hr-en.txt"

    single2bitext(source, target, merge_passive, merge_not_passive)

