import email
import re
import imghdr

# 소문자로 키워드 리스트 작성
keywords = ["free", "sale", "buy", "now", "click", "discount", "offer", "save", "money", "cash", "prize", "guaranteed", "guaranted", "win", "congratulations", "viagra", "sex", "http", "men", "price", "adobe", "cialis"]

# 정규 표현식 패턴
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def extract_email_text(msg):
    email_text = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            email_text += part.get_payload()
    return email_text

def check_email_info(path, keywords):
    with open(path, "r", errors="ignore") as f:
        msg = email.message_from_file(f)

    spam_detected = False  # 스팸 메일 여부를 나타내는 플래그
    email_text = extract_email_text(msg)
    
     # To 헤더 체크
    to_header = msg.get('To', '')
    if '<' not in to_header and '>' not in to_header:
        spam_detected = True
        
    # 여러 개의 키워드 검색
    for keyword in keywords:
        if keyword in email_text.lower():
            spam_detected = True  # 스팸 감지됨
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

def calculate_accuracy(index_file_path, base_email_path):
    total_count = 0
    correct_count = 0

    with open(index_file_path, 'r') as index_file:
        for line in index_file:
            label, email_relative_path = line.strip().split()
            email_path = base_email_path + email_relative_path.replace("../data/", "")
            expected_label = label  # 'spam' 또는 'ham'

            # check_email_info 함수를 사용하여 스팸 여부를 판별
            predicted_label = check_email_info(email_path, keywords)  # 'spam' 또는 'ham'

            if predicted_label == expected_label:
                correct_count += 1
            
            total_count += 1

    accuracy = correct_count / total_count
    return accuracy

if __name__ == "__main__":
    index_file_path = "/mnt/ssd/trec07p/full/index_100"
    base_email_path = "/mnt/ssd/trec07p/data_100/"
    accuracy = calculate_accuracy(index_file_path, base_email_path)
    print(f"정확도: {accuracy * 100}%")
