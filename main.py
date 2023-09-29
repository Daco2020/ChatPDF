__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from typing import Any
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
import streamlit as st
import tempfile
import os


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.text += token
        self.container.markdown(self.text)


from streamlit_extras.buy_me_a_coffee import button

# 후원 버튼
button(username="daco2020", floating=True, width=221)

# Title
st.title("ChatPDF")
st.write("---")

# openai api key input
openai_key = st.text_input("OpenAI API Key", type="password")
st.write("---")

# File Uploader
uploaded_file: UploadedFile = st.file_uploader("Upload PDF file", type="pdf")
st.write("---")


def pdf_to_document(uploaded_file: UploadedFile):
    temp_dir = tempfile.TemporaryDirectory()  # 임시 폴더 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(pages)

    # Embeddings
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load in to chroma
    # 가공된 문서를 임베딩 모델로 저장
    # 임베딩은 정보를 가져오는 역할
    # 생성은 llm 의 역할
    # db = Chroma.from_documents(docs, embeddings_model, persist_directory="chroma_db")
    db = Chroma.from_documents(docs, embeddings_model)

    # Question
    # q = "What is the meaning of life?"
    # llm = ChatOpenAI(temperature=1, streaming=True)
    # retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)

    # 관련 문서 가져오기
    # docs = retriever_from_llm.get_relevant_documents(query=q)

    # 질문 답변
    st.header("PDF에게 질문해보세요")
    q = st.text_input("질문을 입력하세요")

    if st.button("전송"):
        with st.spinner("잠시만 기다려주세요..."):
            chat_box = st.empty()
            stream_handler = StreamHandler(container=chat_box)
            llm = ChatOpenAI(
                openai_api_key=openai_key,
                model_name="gpt-3.5-turbo",
                temperature=0.5,
                streaming=True,
                callbacks=[stream_handler],
            )
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": q})
            st.write(result["result"])
