import streamlit as st
import os
from bedrock_rag import index_documents, query_rag

st.set_page_config(page_title="AWS Bedrock RAG 데모", layout="centered")
st.title("AWS Bedrock RAG 데모")

# 문서 인덱싱 (최초 1회만)
if not os.path.exists("faiss_index/index.faiss"):
    with st.spinner("문서 인덱싱 중..."):
        index_documents("example_docs")
    st.success("문서 인덱싱이 완료되었습니다.")

st.write("질문을 입력하면 RAG 시스템이 답변합니다.")

question = st.text_input("질문을 입력하세요:")
use_graph = st.checkbox("LangGraph 기반 RAG 사용", value=False)

if st.button("질문하기") and question:
    with st.spinner("답변 생성 중..."):
        try:
            answer = query_rag(question, use_graph=use_graph)
            st.markdown(f"**답변:** {answer}")
        except Exception as e:
            st.error(f"오류 발생: {e}") 