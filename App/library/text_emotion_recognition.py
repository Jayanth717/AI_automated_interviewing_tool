from gensim.models import KeyedVectors
from gensim.models import word2vec

import numpy as np
import pandas as pd
import re
import datetime
from operator import itemgetter
from random import randint
import seaborn as sns

import os
import time
import string
import dill
import pickle

from nltk import *
from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords as sw, wordnet as wn
from nltk.stem.snowball import SnowballStemmer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split as tts
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf

# ✅ Use tf.keras everywhere (prevents keras/tf-keras mismatch)
from tensorflow.keras.preprocessing.text import Tokenizer

# ✅ FIX: pad_sequences import (modern + fallback)
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    from keras_preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, LSTM, SpatialDropout1D, Activation, Conv1D, MaxPooling1D, Input, concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


class predict:

    def __init__(self):
        self.max_sentence_len = 300
        self.max_features = 300
        self.embed_dim = 300
        self.NLTKPreprocessor = self.NLTKPreprocessor()


    class NLTKPreprocessor(BaseEstimator, TransformerMixin):
        """
        Transforms input data by using NLTK tokenization, POS tagging
        lemmatization and vectorization.
        """

        def __init__(self, max_sentence_len=300, stopwords=None, punct=None, lower=True, strip=True):
            self.lower = lower
            self.strip = strip
            self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
            self.punct = set(punct) if punct else set(string.punctuation)
            self.lemmatizer = WordNetLemmatizer()
            self.max_sentence_len = max_sentence_len

        def fit(self, X, y=None):
            return self

        def inverse_transform(self, X):
            return X

        def transform(self, X):
            print(str(X))
            output = np.array([(self.tokenize(doc)) for doc in X])
            return output

        def tokenize(self, document):
            lemmatized_tokens = []

            # Clean the text
            document = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", document)
            document = re.sub(r"what's", "what is ", document)
            document = re.sub(r"\'s", " ", document)
            document = re.sub(r"\'ve", " have ", document)
            document = re.sub(r"can't", "cannot ", document)
            document = re.sub(r"n't", " not ", document)
            document = re.sub(r"i'm", "i am ", document)
            document = re.sub(r"\'re", " are ", document)
            document = re.sub(r"\'d", " would ", document)
            document = re.sub(r"\'ll", " will ", document)
            document = re.sub(r"(\d+)(k)", r"\g<1>000", document)

            for sent in sent_tokenize(document):
                for token, tag in pos_tag(wordpunct_tokenize(sent)):

                    token = token.lower() if self.lower else token
                    token = token.strip() if self.strip else token
                    token = token.strip('_') if self.strip else token
                    token = token.strip('*') if self.strip else token

                    if token in self.stopwords or all(char in self.punct for char in token):
                        continue

                    lemma = self.lemmatize(token, tag)
                    lemmatized_tokens.append(lemma)

            doc = ' '.join(lemmatized_tokens)
            tokenized_document = self.vectorize(np.array(doc)[np.newaxis])
            return tokenized_document

        def vectorize(self, doc):
            save_path = "Models/padding.pickle"
            with open(save_path, 'rb') as f:
                tokenizer = pickle.load(f)

            doc_pad = tokenizer.texts_to_sequences(doc)
            doc_pad = pad_sequences(
                doc_pad,
                padding='pre',
                truncating='pre',
                maxlen=self.max_sentence_len
            )
            return np.squeeze(doc_pad)

        def lemmatize(self, token, tag):
            tag = {
                'N': wn.NOUN,
                'V': wn.VERB,
                'R': wn.ADV,
                'J': wn.ADJ
            }.get(tag[0], wn.NOUN)

            return self.lemmatizer.lemmatize(token, tag)


    class MyRNNTransformer(BaseEstimator, TransformerMixin):
        """
        Transformer allowing our Keras model to be included in our pipeline
        """
        def __init__(self, classifier):
            self.classifier = classifier

        def fit(self, X, y):
            batch_size = 32
            num_epochs = 35
            self.classifier.fit(X, y, epochs=num_epochs, batch_size=batch_size, verbose=2)
            return self

        def transform(self, X):
            self.pred = self.classifier.predict(X)
            self.classes = [[0 if el < 0.2 else 1 for el in item] for item in self.pred]
            return self.pred


    def run(self, X, model_name):
        """
        Returns the predictions from the pipeline including our NLTKPreprocessor and Keras classifier.
        """
        def build(classifier):
            model = Pipeline([
                ('preprocessor', self.NLTKPreprocessor),
                ('classifier', classifier)
            ])
            return model

        save_path = 'Models/'

        with open(save_path + model_name + '.json', 'r') as json_file:
            classifier = model_from_json(json_file.read())

        classifier.load_weights(save_path + model_name + '.h5')
        classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model = build(self.MyRNNTransformer(classifier))
        y_pred = model.transform([X])

        K.clear_session()
        return y_pred
