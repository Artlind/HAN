"""
Network
"""
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F

softmax = nn.Softmax(dim=1)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def sentencesplit(doc):
    """
    We separate sentences, '?' and '!' are considered as words
    doc (str): string of sentences we want to split
    output (list) : list of the sentences ["s1", "s2",...]
    """
    out = doc
    out = out.replace("? ", "?.")
    out = out.replace("! ", "!.")
    out = out.split(".")
    i = 0
    while "" in out or " " in out:
        if out[i] == "" or out[i] == " ":
            out.pop(i)
            continue
        i += 1
    return out

def cleantxt(text):
    """
    Gets rid of some useless characters in text
    text (str)
    output (str) : cleaned text
    """
    return ((text.replace(',', '')).replace('/', ' ')).replace('-', ' ')

def clean_sentences(sentences_raw):
    """
    Cleans sentences from sentencesplit to get rid of some useless sentences
    """
    out = []
    for sentence in sentences_raw:
        if sentence.split() != []:
            out.append(sentence)
    return out

def new_parameter(*size):
    """
    Prepares the attention layer for our network
    *size : tuple of the size of the attention vector
    """
    out = torch.nn.Parameter(torch.FloatTensor(*size).to(DEVICE))
    torch.nn.init.xavier_normal_(out)
    return out

class HAN(nn.Module):
    """
    Our classifier
    """
    def __init__(self, embedding_dim, vocab_size, batch_size,
                 number_cat, device, embedd_dict):
        """
        embedding_dim (int): dimension for the embedding matrix
        vocab_size (int) : total number of different words
        batch_size (int)
        number_cat (int) : number of categories
        device (torch.device)
        embedd_dict (dict) : dictionnary assigning an int to every word of our vocabulary
        """
        super(HAN, self).__init__()
        self.embedd_dict = embedd_dict
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.dim_gru = embedding_dim//2
        self.number_cat = number_cat
        self.hidden_gru_words = self.init_hidden_words()
        self.hidden_gru_sentences = self.init_hidden_sentences()

# =============================================================================
# Couches mots
# =============================================================================

        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim).to(self.device)
        self.gru_word = nn.GRU(self.embedding_dim, self.dim_gru,
                               num_layers=1, bias=True, batch_first=True,
                               dropout=0, bidirectional=True).to(self.device)
        self.MLP_word = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.attention_word = new_parameter(self.embedding_dim, 1).to(self.device)

# =============================================================================
# Couches phrases
# =============================================================================

        self.gru_sentence = nn.GRU(self.embedding_dim, self.dim_gru,
                                   num_layers=1, bias=True, batch_first=True,
                                   dropout=0, bidirectional=True).to(self.device)
        self.MLP_sentence = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.attention_sentence = new_parameter(self.embedding_dim, 1).to(self.device)
        self.MLP_classification = nn.Linear(self.embedding_dim, self.number_cat).to(self.device)

    def init_hidden_words(self):
        return torch.zeros(2, 1, self.dim_gru).to(self.device)

    def init_hidden_sentences(self):
        return torch.zeros(2, 1, self.dim_gru).to(self.device)

    def cat(self, text):
        """
        Gives a category to text
        text (str)
        output(int) id of the category that is assigned to text
        """
        a = self.forward([text])
        number = list(max(softmax(a))).index(max(list(max(softmax(a)))))
        return number

# =============================================================================
# FORWARD
# =============================================================================

    def forward(self, doc):
        """
        Forward of the network
        doc (list) : input batch [text1,text2,...]
        output (tensor): tensor which softmax can be seen as a probability vector for
        the input text to be part of each category
        """
        out = torch.tensor([]).float().to(self.device)

        for i in range(len(doc)):
            sentences_raw = sentencesplit(cleantxt(doc[i]))
            sentences_ready = torch.tensor([]).float().to(self.device)
            for sentence in sentences_raw:
                sentence = sentence.split()
                if sentence == []:
                    continue
                lookup_tensor = torch.tensor([]).long().to(self.device)
                for word in sentence:
                    if word in self.embedd_dict:
                        lookup_tensor = torch.cat((lookup_tensor,
                                                   torch.LongTensor([self.embedd_dict[word]])), 0)
                    else:
                        lookup_tensor = torch.cat((lookup_tensor, torch.LongTensor([0])), 0)
                # Word embedding
                xw = self.word_embedding(lookup_tensor).view(1, -1, self.embedding_dim).to(self.device)
                # Word GRU
                self.hidden_gru_words = self.init_hidden_words()
                hw, self.hidden_gru_words = self.gru_word(xw, self.hidden_gru_words)
                # Word MLP
                uw = nn.Tanh()(self.MLP_word(hw)).to(self.device)
                # Word attention
                attention_score = torch.matmul(uw, self.attention_word).squeeze().to(self.device)
                attention_score = F.softmax(attention_score, dim=0).view(uw.size(0), uw.size(1), 1).to(self.device)
                scored_x = (hw * attention_score).to(self.device)
                s = torch.sum(scored_x, dim=1).to(self.device)
                #collecting sentences
                sentences_ready = torch.cat((sentences_ready, s), 0)
            # Sentence GRU
            if len(sentences_ready) == 0:
                out = torch.cat((out,
                                 torch.randn(1, self.number_cat).to(self.device)), 0).to(self.device)
                continue
            sentences_ready_gru = sentences_ready.view(1, -1, self.embedding_dim).to(self.device)
            self.hidden_gru_sentences = self.init_hidden_sentences()
            hs, self.hidden_gru_sentences = self.gru_sentence(torch.tensor(sentences_ready_gru), self.hidden_gru_sentences)
            # SENTENCE MLP
            us = nn.Tanh()(self.MLP_sentence(hs)).to(self.device)
            # Sentence attention
            attention_score = torch.matmul(us, self.attention_sentence).squeeze().to(self.device)
            attention_score = F.softmax(attention_score, dim=0).view(us.size(0), us.size(1), 1).to(self.device)
            scored_x = (hs * attention_score).to(self.device)
            v = torch.sum(scored_x, dim=1).to(self.device)
            # classification
            p = self.MLP_classification(v).to(self.device)
            out = torch.cat((out, p.float()), 0).float().to(self.device)
        return out
