# Reddit's r/AmItheAsshole Dataset

## Download the dataset

You can download the dataset from Zenodo with the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6791836.svg)](https://doi.org/10.5281/zenodo.6791836)

Download the `reddit.gz` file to your computer, e.g., to `~/Downloads`.

## Starting a MongoDB server

The dataset is a gzip dump, and you will need to restore it. First, you need to run MongoDB and ensure it is running. For example, if you have installed `mongodb-community` on macOS from Homebrew, run

```bash
brew services start mongodb-community
```

On Linux, run

```bash
sudo systemctl start mongod
```

For more information, including how to install and start a database on Windows, check the [documentation](https://www.mongodb.com/docs/manual/installation/).

## Restoring the database

Now that you have MongoDB running, we will host the database by loading the dump. Change to the directory where you downloaded `reddit.gz` to, and run

```bash
mongorestore --db reddit --host=localhost --port=27017 --drop --gzip --archive=reddit.gz
```

You can change the IP address and port of the host. Now you are ready to use the dataset.

## Using PyMongo

First start my making a connection through a client:

```python
>>> from pymongo import MongoClient
>>>
>>> # Change these values if you have done it above
>>> HOST, PORT = "localhost", 27017
>>> client = MongoClient(host=HOST, port=PORT)
>>>
>>> # The database is called reddit
>>> db = client.reddit
```

There are two collections in this dataset, `submissions` (posts) and `comments`. We will explore them now.

### Submissions

These are the *posts* created by OPs. To access them:

```python
>>> subs = db.submissions
```

A typical post looks like this:

```python
>>> # Find a post
>>> subs.find_one()
{'_id': '1fy0bx',
 'author': 'flignir',
 'link_flair_text': 'not the asshole',
 'url': 'http://www.reddit.com/r/AmItheAsshole/comments/1fy0bx/aita_i_like_air_conditioning_and_my_coworkers/',
 'title': 'AItA: I like air conditioning and my coworkers like working half-naked.',
 ...}
```

Some important attributes in a post are summarized in the following table.

|Attribute|Meaning|
|:----|:----|
|`_id`|The unique ID of the post|
|`author`|The unique ID of the OP|
|`url`|The URL of the post|
|`title`|The post's title|
|`selftext`|The post's body text|
|`score`|The score of the post. Equals upvotes minus downvotes.|
|`link_flair_text`|The post's flair|
|`created_utc`|The time when the post was created. In the `datetime` format in Python.|

### Comments

These are similar to the submissions. To access them:

```python
>>> cmts = db.comments
```

A typical comment looks like this:

```python
>>> cmts.find_one()
{'_id': 'cagbfr9',
 'author': 'ail33',
 'body': 'I agree with you, she is kind of bitchy',
 'created_utc': datetime.datetime(2013, 6, 11, 2, 18, 26),
 'parent_id': '1fy0bx',
 ...}
```

Most attributes are the same as in submissions. Some differences are captured in the following table.

|Attribute|Meaning|
|:----|:----|
|`link_id`| The ID of the original post this comment replies to|
|`parent_id`| The ID of the *parent*.  If the comment is top-level\, this ID refers to the original post it replies to. Otherwise\, this ID refers to the parent comment it replies to.|
|`body`| The comment's body text|
|`label`| The judgment (`YTA`| `NTA`| `ESH`| `NAH`| `INFO`) in the comment. If there is none| `label` is an empty string.|

So, a *top-level* comment is one that has `link_id` equal to `parent_id`.