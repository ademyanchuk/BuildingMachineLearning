from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tag import brill, brill_trainer

def backoff_tagger(train_sents, tagger_classes, backoff=None):
     for cls in tagger_classes:
       backoff = cls(train_sents, backoff=backoff)
     return backoff

def word_tag_model(words, tagged_words, limit=200):
    fd = FreqDist(words)
    cfd = ConditionalFreqDist(tagged_words)
    most_freq = (word for word, count in fd.most_common(limit))
    return dict((word, cfd[word].max()) for word in most_freq)

def train_brill_tagger(initial_tagger, train_sents, **kwargs):
     templates = [
       brill.Template(brill.Pos([-1])),
       brill.Template(brill.Pos([1])),
       brill.Template(brill.Pos([-2])),
       brill.Template(brill.Pos([2])),
       brill.Template(brill.Pos([-2, -1])),
       brill.Template(brill.Pos([1, 2])),
       brill.Template(brill.Pos([-3, -2, -1])),
       brill.Template(brill.Pos([1, 2, 3])),
       brill.Template(brill.Pos([-1]), brill.Pos([1])),
       brill.Template(brill.Word([-1])),
       brill.Template(brill.Word([1])),brill.Template(brill.Word([-2])),
       brill.Template(brill.Word([2])),
       brill.Template(brill.Word([-2, -1])),
       brill.Template(brill.Word([1, 2])),
       brill.Template(brill.Word([-3, -2, -1])),
       brill.Template(brill.Word([1, 2, 3])),
       brill.Template(brill.Word([-1]), brill.Word([1])),
      ]
     trainer = brill_trainer.BrillTaggerTrainer(initial_tagger, templates, deterministic=True, trace=True)
     return trainer.train(train_sents, max_rules=1000, min_score=3, **kwargs)
