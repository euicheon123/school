import email
import re
import imghdr
import os
import html2text
import nltk
import string

#week2.py 에서 뽑은 keyword list
keywords = ['http', 'com', 'pills', 'per', 'price', 'adobe', '20mg', 'viagra', 'anatrim', 'products', 'online', 'retail', 'www', 'quality', '100mg',]


# 정규 표현식 패턴
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

DATA_DIR = '/mnt/ssd/trec07p/data/'
LABEL_FILE = '/mnt/ssd/trec07p/full/index'
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


def extract_email_text(msg):
    email_text = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            email_text += part.get_payload()
    return email_text

X_all = load_fname_label() # label 읽기
X_train, X_test = split_dataset(X_all) # 데이터셋 분할

def check_email_info(path):
    with open(path, "r", errors="ignore") as f:
        msg = email.message_from_file(f)

    spam_detected = False  # 스팸 메일 여부를 나타내는 플래그
    email_text = extract_email_text(msg)
    email_word = load_email(path)
    email_word = str(email_word)
    

     # To 헤더 체크
    to_header = msg.get('To', '')
    if '<' not in to_header and '>' not in to_header:
        spam_detected = True
        
         # 여러 개의 키워드 검색
    for keyword in keywords:
        if keyword in email_word.lower():
            spam_detected = True
            break
            
        
    # 이메일 본문에 URL이나 이미지 파일이 포함되어 있는지 검사
    if re.search(url_pattern, email_text):
        spam_detected = True
        
    image_parts = [part for part in msg.walk() if part.get_content_type().startswith('image/')]
    for image_part in image_parts:
        image_data = image_part.get_payload(decode=True)
        image_type = imghdr.what(None, h=image_data)
        if image_type is not None:
            spam_detected = True

    for part in msg.walk():
        if part.get_content_type().startswith("application/") or part.get_content_type().startswith("application/x-msdownload"):
            spam_detected = True

    return "spam" if spam_detected else "ham"

def calculate_accuracy(X_all, X_test, base_email_path):
    tp, fp, fn, tn = 0, 0, 0, 0

    for fname in X_test:                                        #X_test는 Key만 뽑힌 데이터셋
        email_path = os.path.join(base_email_path, fname)
        expected_label = "spam" if X_all[fname] == 1 else "ham"  

        # Use check_email_info function to determine spam or not
        predicted_label = check_email_info(email_path)

        if predicted_label == expected_label and predicted_label == "spam":
            tp +=1
        if predicted_label != expected_label and predicted_label == "spam":
            fp +=1
        if predicted_label == expected_label and predicted_label == "ham":
            tn +=1
        if predicted_label != expected_label and predicted_label == "ham":
            fn +=1

    # Calculate metrics
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print("TP=", tp, "\t", "FN=", fn)
    print("FP=", fp, "\t", "TN=", tn)
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1_score))

if __name__ == "__main__":
    calculate_accuracy(X_all, X_test, DATA_DIR)


