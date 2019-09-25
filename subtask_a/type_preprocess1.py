import os
import sys
import logging
import re
import gensim
import pickle
import jieba
import numpy as np
import pandas as pd
from collections import defaultdict

train_幽默类型 = open("txt文件/幽默类型任务_train.txt", encoding = "utf-8")
test_幽默类型 = open("txt文件/幽默类型_test.txt",encoding = "utf-8")

# train = pd.read_csv(train_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)
# test = pd.read_csv(test_幽默类型, header = None, sep='\t', quoting=3, engine = "python", error_bad_lines=False)

train = pd.read_csv(train_幽默类型, header = None, sep='\t', quoting=3, engine = "python")
test = pd.read_csv(test_幽默类型, header = None, sep='\t', quoting=3, engine = "python")

train.columns = ["ID","Contents","Class"]
test.columns = ["ID","Contents","Class"]

train["Class"] = train["Class"] .replace({"谐音": 1, "谐义": 2, "反转": 3})
test["Class"] = test["Class"] .replace({"谐音": 1, "谐义": 2, "反转": 3})

# print(train["Class"])
# print(test["Class"])
# print(train)
'''
def Content_to_wordlist(Content,strip_all=False):
    while '\n' in Content:
        Content = Content.replace('\n', '')
    while ' ' in Content:
        Content = Content.replace(' ', '')
    if len(Content) > 0:  # 如果句子非空
        if strip_all:
            punctuation = """，。＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟‧﹏"""
            re_punctuation = "[{}]+".format(punctuation)
            Content = re.sub(re_punctuation, "", Content)
        else:
            rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
            Content = rule.sub('', Content)

    return(Content.strip())
'''

def build_data_train_test(data_train, data_test,clean_Content = True, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in range(len(data_train)):
        # print('第%d条， 共%d条' % (i, len(data_train)))
        rev = data_train[i]
        y = train["Class"][i]
        # print("y:", y)
        rev = rev.strip()
        # if clean_Content:
        #     orig_rev = Content_to_wordlist(rev)
        # else:
        orig_rev = ' '.join(rev).lower()
        w = jieba.cut(orig_rev)
        words = set(' '.join(w))
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)

    for i in range(len(data_test)):
        rev = data_test[i]
        rev = rev.strip()
        # if clean_Content:
        #     orig_rev = Content_to_wordlist(rev)
        # else:
        orig_rev = ' '.join(rev).lower()
        w = jieba.cut(orig_rev)
        words = set(' '.join(w))
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    logging.info('unk words: %d' % (unk_words))
    return word_vecs


def get_W(word_vecs, k=300):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((k,))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    '''
    clean_train_Contents = []
    for Content in train["Contents"]:
        clean_train_Contents.append(Content_to_wordlist(Content, strip_all=True))

    clean_test_Contents = []
    for Content in test["Contents"]:
        clean_test_Contents.append(Content_to_wordlist(Content, strip_all=True))
    '''
    revs, vocab = build_data_train_test(train["Contents"], test["Contents"])
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))

    # word2vec GoogleNews
    # model_file = os.path.join('vector', 'GoogleNews-vectors-negative300.bin')
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    # Glove Common Crawl
    model_file = os.path.join('word2vec', 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    w2v = load_bin_vec(model, vocab)
    logging.info('word embeddings loaded!')
    logging.info('num words in embeddings: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v, k=model.vector_size)
    logging.info('extracted index from embeddings! ')

    # pickle_file = os.path.join('pickle', 'vader_movie_reviews_glove.pickle3')
    pickle_file = os.path.join('pickle', 'type_train_val_test1111.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')