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

DATA_DIR = '/mnt/ssd/trec07p/data_100/'
LABEL_FILE = '/mnt/ssd/trec07p/full/index_100'
TRAINING_SET_RATIO = 0.70

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
    with open(path, "r", errors="ignore") as f:
        msg = email.message_from_file(f)
        if not msg:
            return ""
        subject = msg['Subject']
        s = subject if subject else ""
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                s += part.get_payload()
            elif part.get_content_type() == 'text/html':
                html = part.get_payload()
                s += html2text.html2text(html)
        return s

# 데이터셋 나누고
def split_dataset(X_all, y_all):
    return train_test_split(X_all, y_all, train_size=TRAINING_SET_RATIO, random_state=2)

# 결과 출력 
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
        str = load_email(os.path.join(DATA_DIR, fname))
        # 입력 데이터 전체
        X_all.append(str)
        # 출력 데이터 전체
        y_all.append(X_all_dic[fname])
    
    # 데이터 셋 분할
    X_train, X_test, y_train, y_test = split_dataset(X_all, y_all)
    
    # BoW 생성 
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    features = vectorizer.get_feature_names_out()
    print("Total {} words(features)".format(len(features)))
    
    clf = SVC()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    print("\nSupport Vector Classifier:")
    display_results(y_test, y_pred)
    
    clf = MultinomialNB()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    print("\nMultinomial Naive Bayes:")
    display_results(y_test, y_pred)
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    print("\nDecision Tree Classifier:")
    display_results(y_test, y_pred)
    
    clf = RandomForestClassifier()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    print("\nRandom Forest Classifier:")
    display_results(y_test, y_pred)
    
    clf = MLPClassifier()
    clf.fit(X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    print("\nMLP Classifier:")
    display_results(y_test, y_pred)

if __name__ == "__main__":
    main()
