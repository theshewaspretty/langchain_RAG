"""
AWS Bedrock RAG 시스템 예제 스크립트
"""

import os
from bedrock_rag import index_documents, query_rag

# 예제 문서 생성
def create_example_documents():
    """예제 문서 생성"""
    os.makedirs("example_docs", exist_ok=True)
    
    # 예제 문서 1
    with open("example_docs/aws_services.txt", "w") as f:
        f.write("""
        AWS(Amazon Web Services)는 클라우드 컴퓨팅 서비스를 제공하는 플랫폼입니다.
        주요 서비스로는 EC2(가상 서버), S3(스토리지), Lambda(서버리스 컴퓨팅), 
        DynamoDB(NoSQL 데이터베이스) 등이 있습니다.
        AWS Bedrock은 Amazon의 생성형 AI 서비스로, 다양한 기반 모델(FM)에 대한 
        액세스를 제공합니다. Claude, Llama 2, Amazon Titan 등의 모델을 API를 통해 
        사용할 수 있습니다.
        """)
    
    # 예제 문서 2
    with open("example_docs/langchain.txt", "w") as f:
        f.write("""
        LangChain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 
        프레임워크입니다. 문서 로딩, 텍스트 분할, 임베딩, 벡터 저장소 연결, 
        프롬프트 관리 등의 기능을 제공합니다.
        LangChain을 사용하면 RAG(Retrieval-Augmented Generation) 시스템을 
        쉽게 구축할 수 있습니다. RAG는 외부 지식을 검색하여 LLM의 응답을 
        향상시키는 기술입니다.
        """)
    
    # 예제 문서 3
    with open("example_docs/langgraph.txt", "w") as f:
        f.write("""
        LangGraph는 LangChain에서 제공하는 라이브러리로, 복잡한 AI 워크플로우를 
        구축하기 위한 도구입니다. 상태 관리, 조건부 분기, 반복 등의 기능을 통해 
        다단계 AI 시스템을 구현할 수 있습니다.
        LangGraph를 사용하면 대화형 에이전트, 복잡한 추론 체인, 멀티에이전트 시스템 등을 
        구축할 수 있습니다. 특히 RAG 시스템에서 검색-생성-평가 등의 단계를 명확하게 
        구조화할 수 있습니다.
        """)

def main():
    # 1. 예제 문서 생성
    create_example_documents()
    print("예제 문서가 생성되었습니다.")
    
    # 2. 문서 인덱싱
    index_documents("example_docs")
    print("문서 인덱싱이 완료되었습니다.")
    
    # 3. 기본 RAG 체인으로 질문
    question1 = "AWS Bedrock이란 무엇인가요?"
    answer1 = query_rag(question1)
    print(f"\n질문: {question1}")
    print(f"답변: {answer1}")
    
    # 4. LangGraph 기반 RAG로 질문
    question2 = "LangChain과 LangGraph의 차이점은 무엇인가요?"
    answer2 = query_rag(question2, use_graph=True)
    print(f"\n질문: {question2}")
    print(f"답변: {answer2}")

if __name__ == "__main__":
    main()
