from PIL import Image
import matplotlib.pyplot as plt

# 키워드와 이미지 파일 경로를 매핑
keyword_image_map = {
    "인공지능 QR코드": [
        "/mnt/data/스크린샷 2024-08-16 오전 7.11.56.png"
    ],
    "카카오 부트캠프": [
        "/mnt/data/카부캠 자리배석.jpeg"
    ]
}


# 질문과 답변에서 키워드를 탐색하여 이미지를 출력하는 함수
def answer_question_with_images(question, answer):
    images_to_display = []
    found_keywords = set()

    # 질문과 답변에서 키워드 탐색
    for keyword, image_paths in keyword_image_map.items():
        if keyword in question or keyword in answer:
            found_keywords.add(keyword)

    # 발견된 키워드에 해당하는 이미지를 수집
    for keyword in found_keywords:
        images_to_display.extend(keyword_image_map[keyword])

    # 중복된 이미지를 제거
    images_to_display = list(set(images_to_display))

    # 응답 출력
    print("응답:", answer)

    # 이미지를 순차적으로 출력
    if images_to_display:
        for image_path in images_to_display:
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()