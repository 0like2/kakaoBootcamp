import requests
import json
import pandas as pd
import re

# 페이지 메타데이터 추출
def clean_text(text):
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_notion_page(page_id, notion_token):
    # 페이지 ID에 하이픈 추가
    formatted_page_id = f'{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}'

    url = f'https://api.notion.com/v1/pages/{formatted_page_id}'
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    print(f"Request URL: {url}")  # 요청 URL 출력
    print(f"Headers: {headers}")  # 요청 헤더 출력

    response = requests.get(url, headers=headers)

    print(f"Response Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# 페이지 블록(내용) 추출
def get_notion_page_blocks(page_id, notion_token):
    formatted_page_id = f'{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:]}'
    url = f'https://api.notion.com/v1/blocks/{formatted_page_id}/children'
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# 제목 추출
def get_notion_page_title(page_data):
    if 'properties' in page_data:
        for prop in page_data['properties'].values():
            if 'title' in prop:
                if len(prop['title']) > 0:
                    return clean_text(prop['title'][0]['plain_text'])
    return "Untitled"



# 데이터 베이스 크롤링 과정 -> 1. page id 기반에서 블록을 스캔하며 데이터베이스 블록 찾기 -> 2. 데이터 베이스 블록 정보 찾기

# 특정 페이지에서 데이터베이스 블록 찾기
def find_databases_in_page_blocks(page_id, notion_token):
    blocks = get_notion_page_blocks(page_id, notion_token)
    if not blocks:
        return []
    database_ids = []
    for block in blocks['results']:
        if block['type'] == 'child_database':
            database_ids.append(block['id'])
    return database_ids

# 데이터 베이스 정보 크롤링
def get_notion_database_items(database_id, notion_token):
    has_more = True
    start_cursor = None
    items = []

    while has_more:
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        headers = {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        body = {"start_cursor": start_cursor} if start_cursor else {}
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 200:
            result = response.json()
            items.extend(result['results'])
            has_more = result.get('has_more', False)
            start_cursor = result.get('next_cursor', None)
        else:
            print(f"Error querying database: {response.status_code}")
            break

    return items

def extract_all_databases_from_page(page_id, notion_token):
    database_ids = find_databases_in_page_blocks(page_id, notion_token)
    all_texts = []
    for database_id in database_ids:
        database_texts = extract_database_and_page_contents(database_id, notion_token)
        all_texts.extend(database_texts)
    return all_texts


# 데이터베이스 항목 및 페이지 내용 추출 함수
def extract_database_and_page_contents(database_id, notion_token):
    database_items = get_notion_database_items(database_id, notion_token)
    all_texts = []

    for item in database_items:
        item_properties = item.get('properties', {})
        item_title = next(
            (v['title'][0]['plain_text'] for k, v in item_properties.items() if v['type'] == 'title' and v['title']),
            'Untitled')
        item_created_time = item.get('created_time', 'N/A')
        item_last_edited_time = item.get('last_edited_time', 'N/A')

        item_texts = []
        item_texts.append(f"Database Item Title: {item_title}")
        item_texts.append(f"Database Item Created Time: {item_created_time}")
        item_texts.append(f"Database Item Last Edited Time: {item_last_edited_time}")

        for prop_name, prop_value in item_properties.items():
            prop_type = prop_value['type']
            if prop_type == 'title':
                prop_text = prop_value['title'][0]['plain_text'] if prop_value['title'] else ''
            elif prop_type == 'rich_text':
                prop_text = ' '.join([t['plain_text'] for t in prop_value['rich_text']])
            elif prop_type == 'multi_select':
                prop_text = ', '.join([option['name'] for option in prop_value['multi_select']])
            elif prop_type == 'select':
                prop_text = prop_value['select']['name'] if prop_value['select'] else ''
            elif prop_type == 'date':
                prop_text = prop_value['date']['start'] if prop_value['date'] else ''
            else:
                prop_text = str(prop_value.get(prop_type, ''))
            item_texts.append(f"{prop_name}: {prop_text}")

        item_texts.append("---")  # 각 데이터베이스 항목 구분
        all_texts.extend(item_texts)

    return all_texts

def extract_text_from_blocks(blocks, notion_token, depth=0):
    texts = []
    for block in blocks['results']:
        block_type = block['type']
        content = block.get(block_type, {})
        if block_type in ['paragraph', 'heading_1', 'heading_2', 'heading_3', 'bulleted_list_item',
                          'numbered_list_item', 'to_do', 'toggle', 'callout', 'quote']:
            if 'rich_text' in block[block_type] and len(block[block_type]['rich_text']) > 0:
                for text in block[block_type]['rich_text']:
                    extracted_text = "  " * depth + text['plain_text']
                    texts.append(extracted_text)
                    print(extracted_text)  # 추출된 텍스트 즉시 출력
        elif block_type == 'child_page':
            child_page_id = block['id'].replace('-', '')
            child_page_data = get_notion_page(child_page_id, notion_token)
            if child_page_data:
                child_page_title = get_notion_page_title(child_page_data)
                child_page_created_time = child_page_data.get('created_time', 'N/A')
                child_page_last_edited_time = child_page_data.get('last_edited_time', 'N/A')
                child_page_info = [
                    "  " * depth + f"Child Page ID: {child_page_id}",
                    "  " * depth + f"Child Page Title: {child_page_title}",
                    "  " * depth + f"Child Page Created Time: {child_page_created_time}",
                    "  " * depth + f"Child Page Last Edited Time: {child_page_last_edited_time}"
                ]
                texts.extend(child_page_info)
                for info in child_page_info:
                    print(info)  # 자식 페이지 정보 즉시 출력
                child_blocks = get_notion_page_blocks(child_page_id, notion_token)
                if child_blocks:
                    child_texts = extract_text_from_blocks(child_blocks, notion_token, depth + 1)
                    texts.extend(child_texts)

                # 자식 페이지 내의 데이터베이스 찾기
                child_database_ids = find_databases_in_page_blocks(child_page_id, notion_token)
                for db_id in child_database_ids:
                    db_texts = extract_database_and_page_contents(db_id, notion_token)
                    for db_text in db_texts:
                        formatted_text = "  " * (depth + 1) + db_text
                        texts.append(formatted_text)
                        print(formatted_text)  # 데이터베이스 내용 즉시 출력
        elif block_type == 'child_database':
            database_id = block['id'].replace('-', '')
            database_texts = extract_database_and_page_contents(database_id, notion_token)
            for db_text in database_texts:
                formatted_text = "  " * depth + db_text
                texts.append(formatted_text)
                print(formatted_text)  # 데이터베이스 내용 즉시 출력

    return texts

# 크롤링 데이터 csv 파일로 저장
def save_texts_to_csv(texts, filename):
    df = pd.DataFrame(texts, columns=["Text"])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def prepare_data_for_training(texts):
    training_data = []
    for text in texts:
        training_data.append({"prompt": f"Extracted Notion content: {text}\n\n###\n\n", "completion": " "})
    return training_data

# Notion 데이터 준비 함수
def prepare_notion_data(notion_token, page_id):
    # 페이지 데이터 가져오기
    page_data = get_notion_page(page_id, notion_token)
    page_created_time = page_data.get('created_time', 'N/A')
    page_last_edited_time = page_data.get('last_edited_time', 'N/A')

    all_texts = []  # 모든 텍스트를 저장할 리스트

    if page_data:
        page_title = get_notion_page_title(page_data)
        print(f"Page ID: {page_id}")
        print(f"Page Title: {page_title}")
        print(f"Page Created Time: {page_created_time}")
        print(f"Page Last Edited Time: {page_last_edited_time}")

        page_blocks_data = get_notion_page_blocks(page_id, notion_token)
        if page_blocks_data:
            print("\nExtracting Page Content:")
            all_texts = extract_text_from_blocks(page_blocks_data, notion_token)

        # 모든 텍스트를 CSV 파일로 저장
        save_texts_to_csv(all_texts, "notion_all_content.csv")
        print("\nSaved all content to notion_all_content.csv")

    else:
        print(f"Failed to retrieve page data for Page ID: {page_id}")


if __name__ == "__main__":
    notion_token = ''  # API key - 나중엔 숨기기
    page_id = 'd90c68f4c3a940d3acf0cc7b8482832a'  # 페이지 ID
    prepare_notion_data(notion_token, page_id)