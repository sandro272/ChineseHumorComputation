#! /usr/bin/env python

import os
import sys
import logging
import pandas as pd
import numpy as np
import pickle

from util import load_train_data, load_test_data, save_result
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Dense, Input, Conv2D, Reshape, Dropout, MaxPooling2D, Flatten, LSTM, MaxPooling1D, Embedding, Bidirectional, GRU
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from Attention_layer import AttentionM

from vote_classifier import VotingClassifier
from metrics import f1

batch_size = 100
nb_epoch = 50
hidden_dim = 120

kernel_size = 3
nb_filter = 60

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev,= [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']
        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_dev = np.array(y_dev)
    # y_valid = np.array(y_valid)
    # print(X_test.shape)            #(120,536)
    # print(y_dev.shape)             #(1400,3)

    return [X_train, X_test, X_dev, y_train, y_dev,]

def lstm_model():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'type_train_val_test22.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev, = make_idx_data(revs, word_idx_map, maxlen=maxlen)
    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)


    hidden = LSTM(hidden_dim, recurrent_dropout=0.25)(embedded)

    output = Dense(4, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1)
    # model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
    #           callbacks=[checkpointer, early_stopping])
    #
    # y_pred = model.predict(X_test, batch_size=batch_size)
    return model

def stacked_lstm_model():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'type_train_val_test2.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)


    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(embedded)
    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(hidden)

    output = Dense(4, activation='softmax')(hidden)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor="val_loss", patience=16, verbose=1)
    # model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
    #           callbacks=[checkpointer, early_stopping])
    #
    # y_pred = model.predict(X_dev, batch_size=batch_size)
    return model

def attention_lstm_model():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'type_train_val_test2.pickle3')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    enc = Bidirectional(GRU(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(embedded)

    att = AttentionM()(enc)

    fc1_dropout = Dropout(0.25)(att)
    fc1 = Dense(50, activation="relu")(fc1_dropout)
    fc2_dropout = Dropout(0.25)(fc1)

    output = Dense(4, activation='softmax')(fc2_dropout)
    model = Model(inputs=sequence, outputs=output)

    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    # early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])

    # model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
    #           callbacks=[checkpointer, early_stopping])
    # y_pred = model.predict(X_dev, batch_size=batch_size)
    return model

clf1 = KerasClassifier(build_fn=lstm_model, nb_epoch =nb_epoch, verbose=1, batch_size = batch_size,)
clf2 = KerasClassifier(build_fn=stacked_lstm_model, nb_epoch =nb_epoch, verbose=1, batch_size = batch_size,)
clf3 = KerasClassifier(build_fn=attention_lstm_model, nb_epoch =nb_epoch, verbose=1, batch_size = batch_size,)

eclf1 = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)], voting='soft')

