"""
AWS Bedrock RAG 시스템 구현
LangChain과 LangGraph를 활용한 RAG(Retrieval-Augmented Generation) 시스템
"""

import os
from typing import Dict, List, Any, TypedDict
import boto3

# .env 파일에서 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv


# AWS 리전 설정
REGION = "ap-northeast-2"  # 현재 설정된 리전으로 변경

# Bedrock 모델 ID 설정
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"  # 사용 가능한 임베딩 모델
LLM_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"  # 사용 가능한 LLM 모델

def init_bedrock_client():
    """Bedrock 클라이언트 초기화"""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION
    )

def init_embeddings():
    """Bedrock 임베딩 모델 초기화"""
    client = init_bedrock_client()
    return BedrockEmbeddings(
        client=client,
        model_id=EMBEDDING_MODEL_ID
    )

def init_llm():
    """Bedrock LLM 초기화"""
    client = init_bedrock_client()
    return ChatBedrock(
        client=client,
        model_id=LLM_MODEL_ID
    )

def load_documents(directory_path: str) -> List[Document]:
    """지정된 디렉토리에서 문서 로드"""
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    return loader.load()

def split_documents(documents: List[Document]) -> List[Document]:
    """문서를 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embeddings):
    """벡터 스토어 생성"""
    return FAISS.from_documents(documents, embeddings)

def create_rag_chain(vector_store, llm):
    """RAG 체인 생성"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 프롬프트 템플릿 정의
    template = """
    <context>
    {context}
    </context>
    
    사용자 질문: {question}
    
    위의 컨텍스트만 사용하여 사용자의 질문에 답변해주세요. 컨텍스트에 관련 정보가 없으면, '제공된 정보로는 답변할 수 없습니다'라고 말해주세요.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # RAG 체인 구성
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# RAG 워크플로우 상태 스키마 정의
class RAGState(TypedDict):
    question: str
    context: list  # context는 문서 리스트
    answer: str

def create_rag_graph():
    """LangGraph를 사용한 RAG 그래프 생성"""
    
    # 노드 함수 정의
    def retrieve(state):
        """문서 검색"""
        embeddings = init_embeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(state["question"])
        return {"question": state["question"], "context": docs}
    
    def generate_answer(state):
        """답변 생성"""
        llm = init_llm()
        
        context_str = "\n\n".join([doc.page_content for doc in state["context"]])
        messages = [
            SystemMessage(content="주어진 컨텍스트를 기반으로 질문에 답변하세요."),
            HumanMessage(content=f"컨텍스트: {context_str}\n\n질문: {state['question']}")
        ]
        
        answer = llm.invoke(messages).content
        return {"question": state["question"], "context": state["context"], "answer": answer}
    
    def should_follow_up(state):
        """후속 질문 필요 여부 결정"""
        if "추가 정보가 필요합니다" in state["answer"]:
            return "follow_up"
        return "end"
    
    # 그래프 구성
    workflow = StateGraph(RAGState)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_answer)
    
    # 엣지 추가
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        should_follow_up,
        {
            "follow_up": "retrieve",  # 추가 정보 필요시 다시 검색
            "end": END  # 아니면 종료
        }
    )
    
    # 시작 노드 설정
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()

def index_documents(docs_directory: str, index_name: str = "faiss_index"):
    """문서 인덱싱 및 벡터 스토어 저장"""
    # 문서 로드 및 분할
    documents = load_documents(docs_directory)
    chunks = split_documents(documents)
    
    # 임베딩 및 벡터 스토어 생성
    embeddings = init_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    
    # 로컬에 벡터 스토어 저장
    vector_store.save_local(index_name)
    
    return vector_store

def query_rag(question: str, use_graph: bool = False):
    """RAG 시스템에 질문"""
    if use_graph:
        # LangGraph 기반 RAG 사용
        graph = create_rag_graph()
        state = {"question": question}
        result = graph.invoke(state)
        return result["answer"]
    else:
        # 기본 RAG 체인 사용
        embeddings = init_embeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = init_llm()
        rag_chain = create_rag_chain(vector_store, llm)
        return rag_chain.invoke(question)

if __name__ == "__main__":
    # 예제 사용법
    # 1. 문서 인덱싱
    # index_documents("path/to/your/documents")
    
    # 2. 질문하기
    # answer = query_rag("당신의 질문을 여기에 입력하세요")
    # print(answer)
    
    # 3. LangGraph 기반 RAG 사용
    # answer = query_rag("당신의 질문을 여기에 입력하세요", use_graph=True)
    # print(answer)
    pass
