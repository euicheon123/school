import os
import email
import html2text
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import re
import numpy as np 
import imghdr


DATA_DIR = '/mnt/ssd/trec07p/data/'
LABEL_FILE = '/mnt/ssd/trec07p/full/index'
TRAINING_SET_RATIO = 0.70

# list of spam-indicating words
spam_indicator_words = ['pills', 'per', 'price', '20mg', 'viagra', 'anatrim', '100mg']
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# Label 읽어 들이기
def load_fname_label():     
    X_all_dic = {}
    with open(LABEL_FILE, "r") as f:
        for line in f:
            line = line.strip()
            label, path = line.split()
            fname = path.split('/')[-1]
            X_all_dic[fname] = 1 if label.lower()=='spam' else 0
    return X_all_dic

# 이메일 불러 들이기 
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

def extract_email_text(msg):
    email_text = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            email_text += part.get_payload()
    return email_text


def check_email_info(path):
    with open(path, "r", errors="ignore") as f:
        msg = email.message_from_file(f)

    email_text = extract_email_text(msg)
    email_word = load_email(path)
    email_word = str(email_word)
    
    features = list(0 for i in range(11))
    
     # To 헤더 체크
    to_header = msg.get('To', '')
    if '<' not in to_header and '>' not in to_header:
        features[0] = 1
             
    # 이메일 본문에 URL이나 이미지 파일이 포함되어 있는지 검사
   
    if re.search(url_pattern, email_text):
        features[1] = 1
        
    
    image_parts = [part for part in msg.walk() if part.get_content_type().startswith('image/')]
    for image_part in image_parts:
        image_data = image_part.get_payload(decode=True)
        image_type = imghdr.what(None, h=image_data)
        if image_type is not None:
            features[2] = 1

    
    for part in msg.walk():
        if part.get_content_type().startswith("application/") or part.get_content_type().startswith("application/x-msdownload"):
            features[3] = 1
            
    # 여러 개의 키워드 검색
    for i in range(7):
        if spam_indicator_words[i] in email_word.lower():
            features[3+i] = 1
    
    return features

    
def split_dataset(X_all, y_all):
    return train_test_split(X_all, y_all, train_size=TRAINING_SET_RATIO, random_state=2)

def display_results(y_test, y_pred):
    print('Accuracy {:.3f}'.format(accuracy_score(y_test, y_pred)))
    mat = confusion_matrix(y_test, y_pred)
    print(mat)

def main():
    
    # 레이블 불러와서 저장
    X_all_dic = load_fname_label()  
    X_all = []
    y_all = []
    for fname in list(X_all_dic.keys()):
        features = check_email_info(os.path.join(DATA_DIR, fname))
        # 입력 데이터 전체
        X_all.append(features)
        # 출력 데이터 전체
        y_all.append(X_all_dic[fname])
        
    X_train, X_test, y_train, y_test = split_dataset(X_all, y_all)
    
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)
    
    clf = SVC()
    clf.fit(X_train_array, y_train)
    y_pred = clf.predict(X_test_array)
    print("\nSupport Vector Classifier:")
    display_results(y_test, y_pred)
    
    clf = MultinomialNB()
    clf.fit(X_train_array, y_train)
    y_pred = clf.predict(X_test_array)
    print("\nMultinomial Naive Bayes:")
    display_results(y_test, y_pred)
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train_array, y_train)
    y_pred = clf.predict(X_test_array)
    print("\nDecision Tree Classifier:")
    display_results(y_test, y_pred)
    
    clf = RandomForestClassifier()
    clf.fit(X_train_array, y_train)
    y_pred = clf.predict(X_test_array)
    print("\nRandom Forest Classifier:")
    display_results(y_test, y_pred)
    
    clf = MLPClassifier()
    clf.fit(X_train_array, y_train)
    y_pred = clf.predict(X_test_array)
    print("\nMLP Classifier:")
    display_results(y_test, y_pred)
    
    
if __name__ == "__main__":
    main()
