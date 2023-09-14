import os
import email
import html2text
import nltk
import string

DATA_DIR = '/mnt/ssd/trec07p/data_100/'
LABEL_FILE = '/mnt/ssd/trec07p/full/index_100'
TRAINING_SET_RATIO = 0.70

def load_fname_label():
    X_all = {}
    with open(LABEL_FILE, "r") as f:
        for line in f:
            line = line.strip()
            label, path = line.split()
            fname = path.split('/')[-1]
            # 1 for spam, 0 for ham
            X_all[fname] = 1 if label.lower()=='spam' else 0
    return X_all

def load_email(path):
    # extract text from email
    with open(path, "r", errors="ignore") as f:
        msg = email.message_from_file(f)
        if not msg:
            return ""
        subject = msg['Subject']
        if subject:
            s = subject
        else:
            s = ""
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                s += part.get_payload()
            elif part.get_content_type() == 'text/html':
                html = part.get_payload()
                s += html2text.html2text(html)
    # tokenize
    tokens = nltk.word_tokenize(s)

    # remove punctuations
    punctuation_list = list(string.punctuation)
    tokens = [ t for t in tokens if t not in punctuation_list ]

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [ t for t in tokens if t not in stopwords ]

    # return a list of tokens
    return tokens

# 데이터셋을 training data와 test data으로 분할
# dataset: dictionary 타입 (filename, label)
def split_dataset(dataset):
    filelist = list(dataset.keys())
    train_data = filelist[ :int(len(filelist)*TRAINING_SET_RATIO) ]
    test_data = filelist[ int(len(filelist)*TRAINING_SET_RATIO): ]
    return train_data, test_data

def make_blacklist(X_all, X_train):
# initialize sets
    spam_words = set()
    ham_words = set()
# 이메일에서 단어를 추출하여, spam_words 또는 ham_words에 추가
    for fname in X_train:
        label = X_all[fname]
        words = load_email(os.path.join(DATA_DIR, fname))
        if label == 1:
            spam_words.update(words)
        else:
            ham_words.update(words)
# blacklist: set of words found only in spam mails
    return spam_words - ham_words

def main():
    X_all = load_fname_label() # label 읽기
    X_train, X_test = split_dataset(X_all) # 데이터셋 분할
    blacklist = make_blacklist(X_all, X_train) # blacklist 생성

    tp, fp, fn, tn = 0, 0, 0, 0
    for fname in X_test:
        label = X_all[fname]
        words = set(load_email( os.path.join(DATA_DIR, fname) ))
        spam_words = blacklist & words
        if spam_words and label==1: # predicted as spam, actually spam
            tp += 1
        if spam_words and label==0: # predicted as spam, actually ham
            fp += 1
        if not spam_words and label==1: # predicted as ham, actually spam
            fn += 1
        if not spam_words and label==0: # predicted as ham, actually ham:
            tn += 1

    total = tp + fp + fn + tn
    print("TP=", tp, "\t", "FN=", fn)
    print("FP=", fp, "\t", "TN=", tn)
    print("TP={:.4f} \t FN={:.4f}".format(tp/total, fn/total))
    print("FP={:.4f} \t TN={:.4f}".format(fp/total, tn/total))

if __name__ == "__main__":
    main()
