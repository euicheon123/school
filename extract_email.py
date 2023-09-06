import email
import re
import imghdr  # 이미지 형식을 확인하기 위한 라이브러리

# 소문자로 키워드 리스트 작성
keywords = ["free", "sale", "buy", "now", "click", "discount", "offer", "save", "money", "cash", "prize", "guaranteed", "win", "congratulations", "viagra", "sex"]

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

    print('Subject --->', msg['Subject'])
    print('From --->', msg['From'])

    # 'To' 헤더 파싱
    to_header = msg['To']
    print('To --->', to_header)

    # 'Date' 헤더 파싱
    date_header = msg['Date']
    print('Date --->', date_header)

    # 메시지 본문 파싱
    spam_detected = False  # 스팸 메일 여부를 나타내는 플래그

    email_text = extract_email_text(msg)

    # 여러 개의 키워드 검색
    for keyword in keywords:
        if keyword in email_text.lower():
            print(f"메시지 본문에 '{keyword}' 키워드가 포함되어 있습니다.")
            spam_detected = True  # 스팸 감지됨

    # 이메일 본문에 URL이나 이미지 파일이 포함되어 있는지 검사
    if re.search(url_pattern, email_text):
        print("이메일 본문에 URL이 포함되어 있습니다.")
        spam_detected = True

    # 이미지 파일을 확인하고 감지
    image_parts = [part for part in msg.walk() if part.get_content_type().startswith('image/')]
    for image_part in image_parts:
        image_data = image_part.get_payload(decode=True)
        image_type = imghdr.what(None, h=image_data)
        if image_type is not None:
            print(f"이메일에 이미지 파일({image_type})이 첨부되어 있습니다.")
            spam_detected = True

    # 링크 감지
    links = re.findall(url_pattern, email_text)
    if links:
        print("이메일에 다음 링크가 포함되어 있습니다:")
        for link in links:
            print(link)

    if spam_detected:
        print("스팸 메일이 의심됩니다.")

if __name__ == "__main__":
    file_number = input("inmail 파일 번호를 입력하세요: ")
    file_path = "/mnt/ssd/trec07p/data/inmail." + file_number

    check_email_info(file_path, keywords)  # keywords 리스트를 함수에 전달
