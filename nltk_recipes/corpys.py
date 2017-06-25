from nltk.corpus.reader import PlaintextCorpusReader
from nltk.corpus.reader.util import StreamBackedCorpusView

class IgnoreHeadingCorpusView(StreamBackedCorpusView):
 def __init__(self, *args, **kwargs):
   StreamBackedCorpusView.__init__(self, *args, **kwargs)
   # open self._stream
   self._open()
   # skip the heading block
   self.read_block(self._stream)
   # reset the start position to the current position in the stream
   self._filepos = [self._stream.tell()]

class IgnoreHeadingCorpusReader(PlaintextCorpusReader):
 CorpusView = IgnoreHeadingCorpusView
