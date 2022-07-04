import os
from .text_utils import Embedding, BOW, TFIDF, Empath, SentenceRoberta
import numpy as np
import pickle
import pandas as pd


class TopicModel:
    def __init__(self, n_topics=5, embedding_model=None):
        self._topic_word_dist = None
        self._topic_posteriors = None
        self._n_topics = n_topics
        self._embedding_model = embedding_model

    def predict(self, texts=[[""]], soft=True):
        return None

    def topic_top_words(self, n_words):
        """
        Return a list of lists. Each inner list contains n_words top words for each topic.
        """
        return None


class LDA(TopicModel):
    def __init__(self, texts=[[""]], n_topics=5, embedding_model=None, **kwargs):
        from sklearn.decomposition import LatentDirichletAllocation
        super().__init__(n_topics=n_topics, embedding_model=embedding_model)
        self._model = LatentDirichletAllocation(n_components=n_topics,
                                                **kwargs)
        self._topic_posteriors = self._model.fit_transform(
            self._embedding_model._embedding)
        self._hard_clusters = np.argmax(self._topic_posteriors, axis=1)

        self._topic_word_dist = self._model.components_
        self._vocab = self._embedding_model._vocab
        self._dim = self._embedding_model._dim
        self._embedding = self._embedding_model._embedding

    def predict(self, texts=[[""]], soft=True):
        embedding = self._embedding_model.embed(texts)
        posteriors = self._model.transform(embedding)
        if not soft:
            # Assign each document to its highest-ranked topic
            hard_clusters = np.argmax(posteriors, axis=1)
            return hard_clusters
        return posteriors

    def topic_top_words(self, n_words=10):
        top_words_all = []
        for word_dist in self._topic_word_dist:
            top_word_idx = np.argsort(word_dist)[::-1][:n_words]
            top_words = [self._vocab[j] for j in top_word_idx]
            top_words_all.append(top_words)
        return top_words_all


nw_model_path = os.path.join("data", "nmf_tfidf_model.pkl")
tfidf_vectorizer_path = os.path.join("data", "tfidf_vectorizer.pkl")
tfidf_emb_path = os.path.join("data", "tfidf_emb.pkl")


class NMF(TopicModel):
    def __init__(self):

        self._embedding_model = TFIDF(vectorizer_path=tfidf_vectorizer_path,
                                      emb_path=tfidf_emb_path)
        self._vocab = self._embedding_model._vocab
        self._dim = self._embedding_model._dim
        self._embedding = self._embedding_model._embedding
        with open(nw_model_path, "rb") as f:
            self._model = pickle.load(f)
        # Turn off verbose mode
        self._model.set_params(verbose=0)
        self._n_topics = self._model.n_components
        self._topic_word_dist = self._model.components_
        self._vocab = self._embedding_model._vocab
        self._dim = self._embedding_model._dim
        self._embedding = self._embedding_model._embedding

    def predict(self, texts=[[""]], soft=True):
        embedding = self._embedding_model.embed(texts)
        posteriors = self._model.transform(embedding)
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)
        if not soft:
            # Assign each document to its highest-ranked topic
            hard_clusters = np.argmax(posteriors, axis=1)
            return hard_clusters
        return posteriors

    def topic_top_words(self, n_words=10):
        top_words_all = []
        for word_dist in self._topic_word_dist:
            top_word_idx = np.argsort(word_dist)[::-1][:n_words]
            top_words = [self._vocab[j] for j in top_word_idx]
            top_words_all.append(top_words)
        return top_words_all

    def posteriors_train(self):
        self._topic_posteriors = self._model.transform(self._embedding)
        # Transform into probabilities: each row sums to 1.0
        self._topic_posteriors /= np.sum(self._topic_posteriors,
                                         axis=1, keepdims=True)
        self._hard_clusters = np.argmax(self._topic_posteriors, axis=1)
        return self._topic_posteriors


lw_model_path = os.path.join("data", "lda_bow_model.pkl")
bow_vectorizer_path = os.path.join("data", "bow_vectorizer.pkl")
bow_emb_path = os.path.join("data", "bow_emb.pkl")


class LDAMerged(TopicModel):
    def __init__(self, n_topics=5, embedding_model=None):
        with open(lw_model_path, "rb") as f:
            self._model = pickle.load(f)
        self._embedding_model = BOW(vectorizer_path=bow_vectorizer_path,
                                    emb_path=bow_emb_path)
        self._vocab = self._embedding_model._vocab
        self._dim = self._embedding_model._dim
        self._embedding = self._embedding_model._embedding

        self._cluster_to_topic_list = ['race', 'breakups', 'education',
                                       'housework', 'wedding', 'driving',
                                       'vacation', 'mental health', 'roommates',
                                       'family', 'technology', 'children',
                                       'death', 'food', 'social media',
                                       'relationships', 'family', 'manners',
                                       'communication', 'pets', 'friends',
                                       'other', 'jokes', 'gaming', 'money',
                                       'family', 'marriage', 'entertainment',
                                       'other', 'restaurant', 'communication',
                                       'relationships', 'school', 'communication',
                                       'driving', 'smoking', 'living',
                                       'relationships', 'drinking', 'appearance',
                                       'living', 'medicine', 'living',
                                       'safety', 'music', 'damage',
                                       'shopping', 'babies', 'money',
                                       'driving', 'parties', 'family',
                                       'communication', 'work', 'religion',
                                       'other', 'phones', 'sex', 'money',
                                       'time', 'gender', 'hygiene', 'family',
                                       'celebrations', 'pets', 'manners',
                                       'sleep', 'wedding', 'appearance', 'food']
        self._topics = list(np.unique(self._cluster_to_topic_list))
        self._n_topics = len(self._topics)
        self._topic_word_dist = np.zeros(
            (self._n_topics, len(self._vocab)))
        self._topic_word_dist = pd.DataFrame(self._topic_word_dist, columns=self._vocab,
                                             index=self._topics)
        for i, row in enumerate(self._model.components_):
            topic_name = self._cluster_to_topic_list[i]
            self._topic_word_dist.loc[topic_name] += row

    def _cluster_to_topic(self, topic_posteriors):
        posteriors = np.zeros((len(topic_posteriors), len(
            self._topics)), dtype=np.float32)
        posteriors = pd.DataFrame(posteriors, columns=self._topics)
        for i in range(len(topic_posteriors[0])):
            topic_name = self._cluster_to_topic_list[i]
            posteriors.loc[:, topic_name] += topic_posteriors[:, i]
        return posteriors

    def predict(self, texts=[[""]]):
        embedding = self._embedding_model.embed(texts)
        posteriors = self._model.transform(embedding)
        posteriors = self._cluster_to_topic(posteriors)
        return posteriors

    def topic_top_words(self, n_words=10):
        words = {}
        for topic, row in self._topic_word_dist.iterrows():
            top_words = row.sort_values(ascending=False).index
            words[topic] = top_words[:n_words].tolist()
        return words

    def posteriors_train(self):
        self._topic_posteriors = self._model.transform(self._embedding)
        self._topic_posteriors = self._cluster_to_topic(self._topic_posteriors)
        return self._topic_posteriors
