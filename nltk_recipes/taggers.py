from nltk.tag import SequentialBackoffTagger
from nltk.corpus import wordnet
from nltk.probability import FreqDist

class WordNetTagger(SequentialBackoffTagger):
     '''
     >>> wt = WordNetTagger()
     >>> wt.tag(['food', 'is', 'great'])
     [('food', 'NN'), ('is', 'VB'), ('great', 'JJ')]
     '''
     def __init__(self, *args, **kwargs):
       SequentialBackoffTagger.__init__(self, *args, **kwargs)

       self.wordnet_tag_map = {
         'n': 'NN',
         's': 'JJ',
         'a': 'JJ',
         'r': 'RB',
         'v': 'VB'
        }
     def choose_tag(self, tokens, index, history):
         word = tokens[index]
         fd = FreqDist()

         for synset in wordnet.synsets(word):
             fd[synset.pos()] += 1
         return self.wordnet_tag_map.get(fd.max())
