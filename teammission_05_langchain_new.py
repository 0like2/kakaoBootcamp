import os
import pandas as pd
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRContextEncoder
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. OpenAI API 키와 모델 이름 설정
OPENAI_API_KEY = "sk-your-api-key-here"  # 여기에 API 키를 설정
MODEL_NAME = "gpt-3.5-turbo"  # 사용할 모델 이름


# 2. 데이터 로드
def load_data():
    df = pd.read_csv('notion_all_content.csv')
    documents = df['content'].tolist()  # Only extract the content column
    return documents


# 3. 문서 분할
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    return splits


# 4. 벡터스토어 생성
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore


# 5. RAG 체인 생성
def create_rag_chain(vectorstore):
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)

    prompt_template = '''아래의 문맥을 사용하여 질문에 답하십시오.
    만약 답을 모른다면, 모른다고 말하고 답을 지어내지 마십시오.
    최대한 세 문장으로 답하고 가능한 한 간결하게 유지하십시오.
    {context}
    질문: {question}
    유용한 답변:'''

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    return qa_chain


# Custom RAG Model Integration
class RAGModel:
    def __init__(self, question_encoder, context_encoder, generation_model, index, tokenizer, documents):
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder
        self.generation_model = generation_model
        self.index = index
        self.tokenizer = tokenizer
        self.documents = documents

    def retrieve(self, question):
        # 질문을 인코딩하여 임베딩으로 변환
        question_inputs = question_tokenizer(question, return_tensors="pt")
        question_outputs = self.question_encoder(**question_inputs)
        question_embedding = question_outputs.pooler_output.detach().numpy()

        # FAISS 인덱스를 사용하여 가장 가까운 문서 검색
        D, I = self.index.search(question_embedding, k=1)  # 가장 가까운 문서 검색
        return self.documents[I[0][0]]

    def generate(self, question, retrieved_doc):
        # 질문과 검색된 문서를 결합하여 입력으로 사용
        input_text = question + " " + retrieved_doc
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")

        # 모델을 통해 결과 출력
        outputs = self.generation_model.generate(inputs, max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def run(self, question):
        # 검색된 문서를 기반으로 답변 생성
        retrieved_doc = self.retrieve(question)
        print(f"Retrieved Document: {retrieved_doc}")
        return self.generate(question, retrieved_doc)


def initialize_rag_model(documents):
    # 문서 인코더와 토크나이저 준비
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # 문서를 임베딩으로 변환
    document_embeddings = []
    for doc in documents:
        inputs = context_tokenizer(doc, return_tensors="pt")
        outputs = context_encoder(**inputs)
        document_embeddings.append(outputs.pooler_output.detach().numpy())

    # 리스트 형태의 임베딩을 하나의 numpy 배열로 변환
    document_embeddings = np.vstack(document_embeddings)

    # FAISS 인덱스 생성 및 임베딩 추가
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)

    # 질문 인코더와 생성 모델 설정
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    generation_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    generation_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

    # RAG 모델 인스턴스 생성
    rag_model = RAGModel(question_encoder, context_encoder, generation_model, index, generation_tokenizer, documents)
    return rag_model


def test_rag_model(question, rag_model):
    answer = rag_model.run(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


# 메인 실행 부분
if __name__ == "__main__":
    documents = load_data()
    splits = split_docs(documents)
    vectorstore = create_vectorstore(splits)
    qa_chain = create_rag_chain(vectorstore)

    # RAG 모델 초기화
    rag_model = initialize_rag_model(documents)

    # RAG 모델 테스트
    test_rag_model("What is the capital of the AI Kingdom?", rag_model)


    # 순수 생성 모델 테스트
    def generate_pure_answer(question):
        inputs = generation_tokenizer.encode(question, return_tensors="pt")
        outputs = generation_model.generate(inputs, max_length=50)
        return generation_tokenizer.decode(outputs[0], skip_special_tokens=True)


    pure_answer = generate_pure_answer("What is the capital of the AI Kingdom?")
    print("Pure Generation Answer:", pure_answer)