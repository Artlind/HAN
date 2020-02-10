"""
Prepares the embedding
"""
from HAN import sentencesplit, cleantxt


def global_dict(data, thd=2):
    """
    Create a pre_embedding dict giving a number to all words
    data has to be a dataframe with the column 'content' containing the texts
    thd (int): words that appear less than thd times are ignored
    """
    out = {}
    compteur = {}
    for text in data['content']:
        for sentence in sentencesplit(cleantxt(text)):
            for word in sentence.split():
                if word not in compteur:
                    compteur[word] = 1
                else:
                    compteur[word] += 1
                    if word not in out and compteur[word] >= thd:
                        out[word] = len(out)+1
    for word in compteur:
        if word not in out:
            out[word] = 0
    return out
