# Topic modeling: Latent Dirichlet Allocation and Nonnegative Matrix Factorization

## Latent Dirichlet Allocation

The topic model we used is LDA on the bag-of-words embedding. Since some clusters have been discarded (we call them the *other* topics), and some clusters are merged together, we have 48 topics left, including 47 *named* topics and 1 *other* topic.

Below are the instructions on setting up this topic model.

### Embedding and pre-trained model: data

1. Create a folder within `scripts` called `data`. Ignore this if it is there already.

2. For the bag-of-words embedding, download the following files and put them in `scripts/data`:
- `bow_emb.pkl`
- `bow_vectorizer.pkl`

3. For the LDA model, download the `lda_bow_model.pkl` and put it in `scripts/data`.

4. Check that there are 3 files in `scripts/data`: `bow_emb.pkl`, `bow_vectorizer.pkl` and `lda_bow_model.pkl`.

### Using the topic model

Our topic model is basically `sklearn`'s `LatentDirichletAllocation`,
plus cluster merging. The class `LDAMerged` within `utils/topic_modeling_utils.py` does just that.

#### Loading the model

Make sure the above `.pkl` files downloaded. Then you can load the model as follows:

```py
>>> from utils.topic_modeling_utils import LDAMerged
>>> model = LDAMerged()
```

#### Getting the vocabulary and topics

To get all tokens in the vocabulary:

```py
>>> len(model._vocab)
10463
>>> model._vocab[:5]
['aa', 'aaa', 'aaron', 'ab', 'abandon']
```

To access the list of named topics:
```py
>>> len(model._n_topics)
48
>>> model._topics
['appearance', 'babies', 'breakups', 'celebrations', 'children', 'communication', 'damage', 'death', 'drinking', 'driving', 'education', 'entertainment', 'family', 'food', 'friends', 'gaming', 'gender', 'housework', 'hygiene', 'jokes', 'living', 'manners', 'marriage', 'medicine', 'mental health', 'money', 'music', 'other', 'parties', 'pets', 'phones', 'race', 'relationships', 'religion', 'restaurant', 'roommates', 'safety', 'school', 'sex', 'shopping', 'sleep', 'smoking', 'social media', 'technology', 'time', 'vacation', 'wedding', 'work']
```

Each topic is described by its most salient words. To get the `n_words` most salient words for each topic:
```py
>>> top_words = model.topic_top_words(n_words=5)
>>> # Note: the list below presents words in decreasing order of salience
>>> top_words["technology"]
['email', 'computer', 'laptop', 'report', 'internet']
```

#### Predicting a post

To predict the topic of a post (or multiple posts):
```py
>>> # This should be a list of strings
>>> posts = ["I was with a friend and I ate his good. AITA?",
             "WIBTA if I broke up with my gf on her birthday?"] 
>>> model.predict(posts)
   appearance    babies  breakups  celebrations  children  communication    damage     death  ...     sleep   smoking  social media  technology      time  vacation   wedding      work
0    0.007143  0.003571  0.003571      0.003571  0.003571       0.014286  0.003571  0.003571  ...  0.003571  0.003571      0.003571    0.003571  0.003571  0.003571  0.007143  0.003571
1    0.007143  0.003571  0.290593      0.254168  0.003571       0.014286  0.003571  0.003571  ...  0.003571  0.003571      0.003571    0.003571  0.003571  0.003571  0.007143  0.003571
```

#### Accessing the training data

`model` was trained on 102,998 AITA posts. To save you time, you can access the posterior probabilities of all these posts $p(k \mid d)$ where $k$ is a topic and $d$ is a document. To find these posteriors:
```py
>>> posteriors = model.posteriors_train()
```
The output is the same as predicting posts above. It's a `pd.DataFrame` with 102,998 rows and 48 columns. To identify what post each row is, we do not have the post IDs directly. You can download the `post_ids.pkl` file on our drive for this. The order matches that in `post_ids.pkl`.

## Nonnegative Matrix Factorization

Another topic model we used is NMF on the TF-IDF embedding. We used 70 latent dimensions.

### Embedding and pre-trained model: data

1. Create a folder within `scripts` called `data`. Ignore this if it is there already.

2. For the TF-IDF embedding, download the following files and put them in `scripts/data`:
- `tfidf_emb.pkl`
- `tfidf_vectorizer.pkl`

3. For the NMF model, download the `nmf_tfidf_model.pkl` and put it in `scripts/data`.

4. Check that there are 3 files in `scripts/data`: `tfidf_emb.pkl`, `tfidf_vectorizer.pkl` and `nmf_tfidf_model.pkl`.

### Using the topic model

This is very similar to `LDAMerged`.

#### Loading the model

```py
>>> from utils.topic_modeling_utils import NMF
>>> model = NMF()
```

#### Getting the vocabulary and topics

To get the vocabulary
```py
>>> len(model._vocab)
10463
>>> model._vocab[:5]
['aa', 'aaa', 'aaron', 'ab', 'abandon']
```

The NMF topics are not named, so we can only access them by their top words. The following method returns a list of 70 lists of top words, each list corresponding to a topic.
```py
>>> top_words = model.topic_top_words(n_words=5)
>>> top_words[0]
['thing', 'time', 'like', 'love', 'try']
```

#### Predicting a post

To predict the topic of a post (or multiple posts):
```py
>>> # This should be a list of strings
>>> posts = ["I was with a friend and I ate his good. AITA?",
             "WIBTA if I broke up with my gf on her birthday?"]
>>> # Returns an array of shape (2, 70). Each row is a topic, and sums to 1.
>>> model.predict(posts)
```

#### Accessing the training data

Posteriors probabilities on the training set:
```py
>>> posteriors = model.posteriors_train()
```