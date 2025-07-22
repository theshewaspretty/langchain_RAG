# AWS Bedrock RAG 시스템 구현 가이드

이 프로젝트는 AWS Bedrock, LangChain, LangGraph를 활용하여 RAG(Retrieval-Augmented Generation) 시스템을 구현하는 방법을 보여줍니다.

## 목차

- [개요](#개요)
- [설치 방법](#설치-방법)
- [AWS 설정](#aws-설정)
- [사용 방법](#사용-방법)
- [주요 기능](#주요-기능)
- [고급 사용법](#고급-사용법)
- [문제 해결](#문제-해결)

## 개요

RAG(Retrieval-Augmented Generation)는 대규모 언어 모델(LLM)의 응답을 외부 지식으로 보강하는 기술입니다. 이 프로젝트에서는 다음 구성 요소를 사용합니다:

- **AWS Bedrock**: 다양한 기반 모델(FM)에 접근할 수 있는 Amazon의 생성형 AI 서비스
- **LangChain**: LLM 애플리케이션 개발을 위한 프레임워크
- **LangGraph**: 복잡한 AI 워크플로우를 구축하기 위한 라이브러리
- **FAISS**: 효율적인 벡터 검색을 위한 라이브러리

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. 환경 변수 설정 (AWS 자격 증명):

AWS CLI가 이미 구성되어 있다면 별도의 설정이 필요 없습니다. 그렇지 않은 경우 다음과 같이 환경 변수를 설정하세요:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_REGION=your_region  # 예: us-east-1
```

또는 `.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

## AWS 설정

1. AWS 계정에서 Bedrock 서비스에 접근 권한이 있는지 확인하세요.
2. 사용하려는 모델(예: Claude, Titan)에 대한 액세스 권한을 요청하세요.
3. IAM 사용자에게 다음 권한이 있는지 확인하세요:
   - `bedrock:InvokeModel`
   - `bedrock:InvokeModelWithResponseStream`

## 사용 방법

### 기본 사용법

1. 문서 인덱싱:

```python
from bedrock_rag import index_documents

# 문서가 있는 디렉토리 지정
index_documents("path/to/your/documents")
```

2. 질문하기:

```python
from bedrock_rag import query_rag

# 기본 RAG 체인 사용
answer = query_rag("당신의 질문을 여기에 입력하세요")
print(answer)

# LangGraph 기반 RAG 사용
answer = query_rag("당신의 질문을 여기에 입력하세요", use_graph=True)
print(answer)
```

### 예제 실행

제공된 예제 스크립트를 실행하여 시스템을 테스트할 수 있습니다:

```bash
python example.py
```

이 스크립트는 예제 문서를 생성하고, 인덱싱한 후, 두 가지 방식(기본 RAG와 LangGraph 기반 RAG)으로 질문에 답변합니다.

## 주요 기능

### 1. 문서 로딩 및 분할

```python
documents = load_documents("path/to/your/documents")
chunks = split_documents(documents)
```

### 2. 임베딩 및 벡터 저장소 생성

```python
embeddings = init_embeddings()
vector_store = create_vector_store(chunks, embeddings)
```

### 3. RAG 체인 구성

```python
llm = init_llm()
rag_chain = create_rag_chain(vector_store, llm)
```

### 4. LangGraph 기반 워크플로우

```python
graph = create_rag_graph()
result = graph.invoke({"question": "당신의 질문"})
```

## 고급 사용법

### 1. 다른 Bedrock 모델 사용

`bedrock_rag.py` 파일에서 모델 ID를 변경하여 다른 Bedrock 모델을 사용할 수 있습니다:

```python
# 임베딩 모델
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

# LLM 모델
LLM_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"  # 다른 모델로 변경 가능
```

### 2. 검색 파라미터 조정

검색 결과의 품질을 향상시키기 위해 검색 파라미터를 조정할 수 있습니다:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",  # 또는 "mmr"
    search_kwargs={"k": 5}  # 검색할 문서 수 증가
)
```

### 3. 커스텀 프롬프트 템플릿

프롬프트 템플릿을 수정하여 응답의 품질을 향상시킬 수 있습니다:

```python
template = """
<context>
{context}
</context>

사용자 질문: {question}

위의 컨텍스트를 기반으로 상세하게 답변해주세요. 답변은 다음 형식으로 구성해주세요:
1. 요약
2. 주요 포인트
3. 추가 정보
"""
```

## 문제 해결

### 일반적인 오류

1. **AWS 자격 증명 오류**:
   - AWS CLI가 올바르게 구성되어 있는지 확인하세요.
   - 환경 변수가 올바르게 설정되어 있는지 확인하세요.

2. **모델 액세스 오류**:
   - AWS Bedrock 콘솔에서 사용하려는 모델에 대한 액세스 권한이 있는지 확인하세요.

3. **메모리 오류**:
   - 대용량 문서를 처리할 때는 청크 크기를 줄이거나 배치 처리를 고려하세요.

### 성능 최적화

1. **임베딩 캐싱**:
   - 반복적인 임베딩 계산을 피하기 위해 임베딩을 캐싱하세요.

2. **청크 크기 조정**:
   - 문서 특성에 따라 청크 크기와 오버랩을 조정하세요.

3. **벡터 저장소 최적화**:
   - 대규모 문서 컬렉션의 경우 FAISS 대신 Pinecone이나 Weaviate와 같은 관리형 벡터 데이터베이스 사용을 고려하세요.
