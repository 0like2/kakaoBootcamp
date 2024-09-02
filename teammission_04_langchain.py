# 1. 환경설정
from dotenv import load_dotenv
import os


def initialize_environment():
    # 환경 변수 로드(API 키 등)
    load_dotenv()

    # 모델 및 캐시 경로 설정
    os.environ["TRANSFORMERS_CACHE"] = "./cache/"
    os.environ["HF_HOME"] = "./cache/"

    # LangSmith 추적 설정(프로젝트 이름 입력)
    logging.langsmith("RAG-Project")
    print("LangSmith 추적을 시작합니다.")


# 2. 데이터 로드 및 전처리
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_preprocess_data(file_path):
    # 문서 로드
    loader = TextLoader(file_path=file_path)

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(loader.load())
    return documents


# 3. 임베딩 생성 및 벡터 저장소 구축
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(documents):
    # 임베딩 모델 설정
    embedding_model = OpenAIEmbeddings()

    # 벡터 저장소 구축
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store


# 4. 검색기 설정
def setup_retriever(vector_store):
    # 검색기 설정
    retriever = VectorStoreRetriever(vectorstore=vector_store)
    return retriever


# 5. 모델 로드
from langchain.chat_models import ChatOpenAI


def load_language_model():
    # ChatOpenAI 객체 생성
    gpt = ChatOpenAI(temperature=0.7, model_name="gpt-4", streaming=True)
    return gpt


# 6. 프롬프트 템플릿 및 결과 파서 설정
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def setup_prompt_and_parser():
    # 프롬프트 템플릿 구성
    prompt_template = PromptTemplate.from_template(
        "Given the following context, answer the question: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    output_parser = StrOutputParser()
    return prompt_template, output_parser


# 7. 체인 구성
from langchain.chains import RetrievalQA


def create_rag_chain(gpt, retriever, output_parser):
    rag_chain = RetrievalQA.from_chain_type(
        llm=gpt,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        output_parser=output_parser
    )
    return rag_chain


# 8. 체인 실행 및 스트리밍 출력
### from langchain_teddynote.messages import stream_response


def generate_answer(rag_chain, question):
    answer = rag_chain({"question": question})
###    stream_response(answer["result"])  # 스트리밍 출력
    return answer


def main():
    # 1. 환경 설정
    initialize_environment()

    # 2. 데이터 로드 및 전처리
    documents = load_and_preprocess_data(file_path="your_document.txt")

    # 3. 임베딩 생성 및 벡터 저장소 구축
    vector_store = create_vector_store(documents)

    # 4. 검색기 설정
    retriever = setup_retriever(vector_store)

    # 5. 모델 로드
    gpt = load_language_model()

    # 6. 프롬프트 템플릿 및 결과 파서 설정
    prompt_template, output_parser = setup_prompt_and_parser()

    # 7. 체인 구성
    rag_chain = create_rag_chain(gpt, retriever, output_parser)

    # 8. 체인 실행 및 스트리밍 출력
    question = "What is the capital of South Korea?"
    generate_answer(rag_chain, question)


if __name__ == "__main__":
    main()