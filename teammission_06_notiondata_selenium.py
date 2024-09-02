# 설정
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd
import json

# 셀레니움 드라이버 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")


# 함수 정의
def find_linktext(driver):
    links = driver.find_elements(By.TAG_NAME, 'a')
    main_text = [link.text for link in links if link.text not in ["", "kakao tech bootcamp mission&quest!"]]
    return main_text


if __name__ == "__main__":
    save_text = []
    driver = webdriver.Chrome(options=chrome_options)

    # 첫 번째 URL 처리
    url = "https://goormkdx.notion.site/kakao-tech-bootcamp-0710eb08b5a743bea83e1871c5ae7465"
    driver.get(url)
    time.sleep(4)
    main_text = find_linktext(driver)
    print(main_text)

    for i in main_text[4:]:
        driver.find_element(By.LINK_TEXT, i).click()
        time.sleep(4)
        page_text = driver.find_element(By.TAG_NAME, "body").text
        save_text.append(page_text)
        driver.back()

    # 두 번째 URL 처리
    url = "https://goormkdx.notion.site/1-dccf703f0a5744d496543b7dac760c20"
    driver.get(url)
    time.sleep(4)
    page_text = driver.find_element(By.TAG_NAME, "body").text
    save_text.append(page_text)
    driver.quit()

    # 추가 URL 처리 및 데이터 저장
    additional_urls = [
        "https://goormkdx.notion.site/PBL-f7069a8070fe40afb01d4b79fb92cdd0",
        "https://goormkdx.notion.site/86660d5567e7426394a370d6a323d1eb",
        "https://goormkdx.notion.site/9ec51d9c2a574cb78ede1a00213812b3",
        "https://goormkdx.notion.site/0bbf635fd7a44be4b880adc173153126",
        "https://goormkdx.notion.site/8668d8a736f44236ab756b1165b729d1"
    ]

    for url in additional_urls:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(4)
        page_text = driver.find_element(By.TAG_NAME, "body").text
        save_text.append(page_text)
        driver.quit()

    # 추가 정보
    save_text.extend([
        "카카오테크 부트캠프 홈페이지: https://kakao-tech-bootcamp.goorm.io/",
        "카카오테크 부트캠프 ZEP: https://zep.us/play/8lj15q",
        "통합신청센터 바로가기: https://forms.gle/egsfisGU4uHeGdM17",
        "구름 EXP 바로가기: https://exp.goorm.io/education/TESIiAqKYZ88RcfzuJ/dashboard#/",
        "디스코드 바로가기: https://discord.gg/gGW8JJ4e",
        "풀스택 실시간 Zoom 강의실: https://zoom.us/j/91064080996?pwd=azFrUHJmby9jZ0t1ZkE1bGtGK2ZKQT09",
        "인공지능 실시간 Zoom 강의실: https://zoom.us/j/94030410356?pwd=eHVMSFZyNDEyS01SWXkzSVVsN1NSZz09",
        "클라우드 실시간 Zoom 강의실: https://us06web.zoom.us/j/88542079125?pwd=VklaifXx0KJMxQkE4cCFb6NEGbhmQH.1"
    ])

    # 저장된 텍스트 CSV로 저장
    df = pd.DataFrame(save_text, columns=["Notion"])
    df.to_csv("output.csv", index=False)

    # 저장된 데이터 확인
    data = pd.read_csv("output.csv")
    print(data)



    print("스크립트 실행이 완료되었습니다.")