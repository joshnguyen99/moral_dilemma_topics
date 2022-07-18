import os
from nltk.corpus import stopwords
import spacy
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, \
    TfidfVectorizer, \
    CountVectorizer
from empath import Empath
import pickle
lexicon = Empath()

# Stop words
STOP_WORDS = stopwords.words("english")
STOP_WORDS = ENGLISH_STOP_WORDS.union(set(STOP_WORDS))
# Remove some common words in AITA
STOP_WORDS = STOP_WORDS.union({'from', 'subject', 're', 'use', "to",
                               'say', 'go', 'tell', 'get',
                               'want', 'know', 'feel', 'ask', 'think',
                               'would', 'say', 'make', 'could', 'give',
                               'take', 'go', 'really', 'think',
                               'may', 'might', 'possibly', "to", "from",
                               'aita', 'wibta', 'be'})

bow_vectorizer_path = os.path.join("data", "bow_vectorizer.pkl")
bow_emb_path = os.path.join("data", "bow_emb.pkl")

tfidf_vectorizer_path = os.path.join("data", "bow_vectorizer.pkl")
tfidf_emb_path = os.path.join("data", "tfidf_emb.pkl")

bert_emb_path = os.path.join("data", "bert_emb.pkl")

empath_emb_path = os.path.join("data", "empath_emb.pkl")


def remove_stopwords(wordlist=[]):
    """
    Remove stop words from a list of words.
    Args:
            wordlist: a list of words
    Return: a list of words in text that are not stop words
    """
    return list(filter(lambda word: word not in STOP_WORDS, wordlist))


# Tokenization. No need for dependency parsing and named entity recognition
try:
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
except Exception:
    import os
    os.system("python3 -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])


def tokenize(text="", allowed_pos_tags=["NOUN", "VERB", "ADJ", "ADP"]):
    """
    Tokenize a string.
    Args:
            text: a string
            allowed_pos_tags: list of allowed spaCy part-of-speech tags
    Return: a list of strings
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc
            if token.pos_ in allowed_pos_tags and
            len(token.lemma_) > 1 and
            token.lemma_.isalpha() and
            token.lemma_ != "-PRON-"]  # remove pronouns


def process_text(text=""):
    """
    Process, tokenize and lemmatize text.
    Args:
        text: a string
    Return: a list of tokens
    """
    # Tokenize and lemmatize
    tokens = tokenize(text)
    # Remove stop words
    tokens = remove_stopwords(tokens)
    return tokens


class Embedding:
    def __init__(self, texts=[""]):
        self._embedding = None
        self._vocab = None

    def embed(self, texts=[""]):
        return None


class BOW(Embedding):
    @staticmethod
    def _bow_embed(texts=[""], filter_extremes=True,
                   max_features=15000, ngram_range=(1, 1),
                   return_embedding=True, verbose=False):
        """
        Create a bag-of-words embedding for a collection of texts.
        Args:
            texts: a list documents
            filter_extremes: boolean, if True then all tokens that appear in fewer
                            than 5 documents, or more than 99% of the documents 
                            will be discarded.
            max_features: the number of top n-grams to retain. If None, all words
                        will remain.
            n_gram_range: a tuple of of (min n-gram size, max n-gram size). 
            return_embedding: boolean, if True then the embeddings of texts will
                            be returned.
        Return:
            bow_vectorizer: sklearn CountVectorizer object.
            embeddings: 2D sparse array for the BoW embeddings of texts. 
                        The shape is (len(texts), number of n-grams). If
                        return_embedding is False, then this is None.
        """
        if filter_extremes:
            min_df, max_df = 20, 1.0
            if max_df * len(texts) < min_df:
                min_df, max_df = 1, 0.9999
        if verbose:
            print("BoW Embedding")

        bow_vectorizer = CountVectorizer(max_df=max_df,
                                         min_df=min_df,
                                         max_features=max_features,
                                         tokenizer=process_text,
                                         ngram_range=ngram_range,
                                         dtype=np.float32)  # save space
        embeddings = None
        if return_embedding:
            embeddings = bow_vectorizer.fit_transform(texts)
        return bow_vectorizer, embeddings

    def __init__(self, texts=[""],
                 vectorizer_path=bow_vectorizer_path,
                 emb_path=bow_emb_path,
                 **kwargs):
        super().__init__(texts)
        if vectorizer_path is not None and emb_path is not None:
            with open("data/bow_vectorizer.pkl", "rb") as f:
                self._vectorizer = pickle.load(f)
            self._vocab = self._vectorizer.get_feature_names()
            self._dim = len(self._vocab)
            with open(emb_path, "rb") as f:
                self._embedding = pickle.load(f)
        else:
            self._vectorizer, self._embedding = self._bow_embed(
                texts=self._texts, **kwargs)
            self._vocab = self._vectorizer.get_feature_names()
            self._dim = len(self._vocab)

    def embed(self, texts=[""]):
        embedding = self._vectorizer.transform(texts)
        return embedding


class TFIDF(Embedding):
    @staticmethod
    def _tfidf_embed(texts=[""], filter_extremes=True,
                     max_features=15000, ngram_range=(1, 1),
                     return_embedding=True, verbose=False):
        """
        Create a TF-IDF embedding for a collection of texts.
        Args:
            texts: a list documents
            filter_extremes: boolean, if True then all tokens that appear in fewer
                            than 5 documents, or more than 99% of the documents 
                            will be discarded.
            max_features: the number of top n-grams to retain. If None, all words
                        will remain.
            n_gram_range: a tuple of of (min n-gram size, max n-gram size). 
            return_embedding: boolean, if True then the embeddings of texts will
                            be returned.
        Return:
            tfidf_vectorizer: the sklearn vectorizer
            embeddings: 2D sparse array for the TF-IDF embeddings of texts. 
                        The shape is (len(texts), number of n-grams). If
                        return_embedding is False, then this is None.
        """
        if filter_extremes:
            min_df, max_df = 20, 1.0
            if max_df * len(texts) < min_df:
                min_df, max_df = 1, 0.9999
        if verbose:
            print("TF-IDF Embedding")
        tfidf_vectorizer = TfidfVectorizer(max_df=max_df,
                                           min_df=min_df,
                                           max_features=max_features,
                                           tokenizer=process_text,
                                           ngram_range=ngram_range,
                                           dtype=np.float32)  # save space
        embeddings = None
        if return_embedding:
            embeddings = tfidf_vectorizer.fit_transform(texts)
        return tfidf_vectorizer, embeddings

    def __init__(self, texts=[""],
                 vectorizer_path=tfidf_vectorizer_path,
                 emb_path=tfidf_emb_path,
                 **kwargs):
        super().__init__(texts)
        if vectorizer_path is not None and emb_path is not None:
            with open(vectorizer_path, "rb") as f:
                self._vectorizer = pickle.load(f)
            self._vocab = self._vectorizer.get_feature_names()
            self._dim = len(self._vocab)
            with open(emb_path, "rb") as f:
                self._embedding = pickle.load(f)
        else:
            self._vectorizer, self._embedding = self._tfidf_embed(
                texts=self._texts, **kwargs)
            self._vocab = self._vectorizer.get_feature_names()
            self._dim = len(self._vocab)

    def embed(self, texts=[""]):
        embedding = self._vectorizer.transform(texts)
        return embedding


class Empath(Embedding):
    @staticmethod
    def empath_embed(texts=[""], normalize=True, return_categories=False,
                     emb_path=empath_emb_path):
        """
        Create an Empath Embedding for a collection of texts.
        Args:
            texts: a list documents
            normalize: boolean, normalize the count of terms by token count
        Returns:
            embeddings: 2D array for the Empath embeddings. The shape is
                        (len(texts), 194).
        """
        categories = sorted(lexicon.cats.keys())
        embedding = np.zeros((len(texts), len(lexicon.cats)), dtype=np.float32)
        for i, text in enumerate(texts):
            try:
                scores = lexicon.analyze(text, normalize=normalize)
                scores = [scores[cat] for cat in categories]
            except RuntimeError:
                scores = [0 for _ in categories]
            embedding[i, :] = scores
        if return_categories:
            return embedding, categories
        return embedding

    def __init__(self, texts=[""],
                 emb_path=empath_emb_path, **kwargs):
        super().__init__(texts)

        if emb_path is not None:
            with open(emb_path, "rb") as f:
                self._embedding = pickle.load(f)
        else:
            self._embedding = self._embed_fn(
                texts=self._texts, **kwargs)

        self._embed_fn = self.empath_embed
        self._vectorizer = None
        self._vocab = sorted(lexicon.cats.keys())
        self._dim = len(self._vocab)

    def embed(self, texts=[""]):
        embedding = self._embed_fn(texts)
        return embedding


class SentenceRoberta(Embedding):
    @staticmethod
    def bert_embed_sentence_transformer(texts, model="stsb-roberta-large",
                                        device="cuda"):
        """
        Create a BERT Embedding for a collection of texts, using the Sentence
        Transformer tool.
        Args:
            texts: a list documents
            model: model name, default to stsb-roberta-large
            device: the device to run BERT on. If run only on CPU, use "cpu".
        Returns:
            embeddings: 2D array for the BERT embeddings. The shape is
                        (len(texts), 1024).
        """
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model)
        model.to(device)
        embeddings = model.encode(texts)
        return embeddings

    def __init__(self, texts=[""],
                 embpath=bert_emb_path,
                 **kwargs):
        super().__init__(texts)
        self._embed_fn = self.bert_embed_sentence_transformer
        self._vectorizer = None
        if embpath is not None:
            with open(embpath, "rb") as f:
                self._embedding = pickle.load(f)
        else:
            self._embedding = self._embed_fn(texts=self._texts, **kwargs)
        self._vocab = None
        self._dim = self._embedding.shape[1]

    def embed(self, texts=[""]):
        embedding = self._embed_fn(texts)
        return embedding
