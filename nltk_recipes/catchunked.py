from nltk.corpus.reader import CategorizedCorpusReader, ChunkedCorpusReader

class CategorizedChunkedCorpusReader(CategorizedCorpusReader,
     ChunkedCorpusReader):
    def __init__(self, *args, **kwargs):
         CategorizedCorpusReader.__init__(self, kwargs)
         ChunkedCorpusReader.__init__(self, *args, **kwargs)
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError('Specify fileids or categories, not both')
        if categories is not None:
            return self.fileids(categories)
        else:
            return fileids

    def raw(self, fileids=None, categories=None):
        return ChunkedCorpusReader.raw(self, self._resolve(fileids, categories))
    def words(self, fileids=None, categories=None):
        return ChunkedCorpusReader.words(self, self._resolve(fileids, categories))
    def sents(self, fileids=None, categories=None):
        return ChunkedCorpusReader.sents(self, self._resolve(fileids, categories))

    def paras(self, fileids=None, categories=None):
        return ChunkedCorpusReader.paras(self, self._resolve(fileids,categories))

    def tagged_words(self, fileids=None, categories=None):
        return ChunkedCorpusReader.tagged_words(self, self._resolve(fileids, categories))
    def tagged_sents(self, fileids=None, categories=None):
        return ChunkedCorpusReader.tagged_sents(self,self._resolve(fileids, categories))
    def tagged_paras(self, fileids=None, categories=None):
        return ChunkedCorpusReader.tagged_paras(self,self._resolve(fileids, categories))


    def chunked_words(self, fileids=None, categories=None):
        return ChunkedCorpusReader.chunked_words(self,self._resolve(fileids, categories))
    def chunked_sents(self, fileids=None, categories=None):
        return ChunkedCorpusReader.chunked_sents(self,self._resolve(fileids, categories))
    def chunked_paras(self, fileids=None, categories=None):
        return ChunkedCorpusReader.chunked_paras(self,self._resolve(fileids, categories))
