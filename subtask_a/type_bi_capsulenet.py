#! /usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging

import pickle
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, CuDNNLSTM, CuDNNGRU, Bidirectional, Input, RepeatVector, Permute, \
    TimeDistributed, SpatialDropout1D, Flatten, GRU
from keras.preprocessing import sequence
from keras.utils import np_utils

from keras.callbacks import EarlyStopping, ModelCheckpoint
from capsule_net import Capsule
from metrics import f1

hidden_dim = 120
batch_size = 1
nb_epoch = 20


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    return x


def make_idx_data(revs: object, word_idx_map: object, maxlen: object = 60) -> object:
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, y_train, y_test, X_dev, y_dev = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)
            y_test.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)

    y_train = np_utils.to_categorical(y_train)
    y_dev = np_utils.to_categorical(y_dev)

    return [X_train, X_test, y_train, y_test, X_dev, y_dev]


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'type_train_val_test2.pickle3')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_test, y_train, y_test, X_dev, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension num of word vector [num_features]: %d" % num_features)

    Routings = 20
    Num_capsule = 60
    Dim_capsule = 120

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W],
                                   trainable=False)(sequence_input)
    embedded_sequences = SpatialDropout1D(0.1)(embedded_sequences)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(embedded_sequences)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings, share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(0.1)(capsule)
    output = Dense(4, activation='softmax')(capsule)
    model = Model(inputs=[sequence_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2,
              shuffle=False, callbacks=[checkpointer, early_stopping])
    y_pred = model.predict(X_dev, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_dev = np.argmax(y_dev, axis=1)
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print(precision_score(y_dev, y_pred, average='macro'))
    print(recall_score(y_dev, y_pred, average='macro'))
    print(accuracy_score(y_dev, y_pred))
    print(f1_score(y_dev, y_pred, average='macro'))


