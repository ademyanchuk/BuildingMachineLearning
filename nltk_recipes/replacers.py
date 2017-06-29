#!/usr/local/bin/python3

import re
from nltk.corpus import wordnet
import enchant
from nltk.metrics import edit_distance
import yaml

replacement_patterns = [
     (r'won\'t', 'will not'),
     (r'can\'t', 'cannot'),
     (r'i\'m', 'i am'),
     (r'ain\'t', 'is not'),
     (r'(\w+)\'ll', '\g<1> will'),
     (r'(\w+)n\'t', '\g<1> not'),
     (r'(\w+)\'ve', '\g<1> have'),
     (r'(\w+)\'s', '\g<1> is'),
     (r'(\w+)\'re', '\g<1> are'),
     (r'(\w+)\'d', '\g<1> would')
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self,text):
        s = text
        for (pattern,repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

class RepeatReplacer(object):
  def __init__(self):
    self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    self.repl = r'\1\2\3'

  def replace(self, word):
    if wordnet.synsets(word):
        return word
    repl_word = self.repeat_regexp.sub(self.repl, word)
    if repl_word != word:
      return self.replace(repl_word)
    else:
      return repl_word

class SpellingReplacer(object):
     def __init__(self, dict_name='en', max_dist=2):
       self.spell_dict = enchant.Dict(dict_name)
       self.max_dist = max_dist
     def replace(self, word):
       if self.spell_dict.check(word):
         return word
       suggestions = self.spell_dict.suggest(word)
       if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
         return suggestions[0]
       else:
         return word

class CustomSpellingReplacer(SpellingReplacer):
     def __init__(self, spell_dict, max_dist=2):
       self.spell_dict = spell_dict
       self.max_dist = max_dist

class WordReplacer(object):
     def __init__(self, word_map):
       self.word_map = word_map
     def replace(self, word):
       return self.word_map.get(word, word)

class YamlWordReplacer(WordReplacer):
     def __init__(self, fname):
       word_map = yaml.load(open(fname))
       super(YamlWordReplacer, self).__init__(word_map)

class AntonymReplacer(object):
    def replace(self, word, pos=None):
        antonyms = set()
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words

class AntonymWordReplacer(WordReplacer, AntonymReplacer):
    pass
    
