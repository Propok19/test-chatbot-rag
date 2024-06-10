from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

from enums.prompt_templates import PromptTemplates

import streamlit as st


class PdfRAG:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
        self.retrievers = {"documents/annualreport2223.pdf": self.init_retriever("documents/annualreport2223.pdf"),
                           "documents/Airbus-Annual-Report-2023.pdf": self.init_retriever("documents/Airbus-Annual-Report-2023.pdf"),
                           }
        self.chat_history = []

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def init_retriever(self, filepath: str):
        db_connection = Chroma(persist_directory=f'./{filepath}_db/', embedding_function=OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"]))
        return db_connection.as_retriever()

    def get_answer(self, question: str, doc: str):
        custom_rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptTemplates.CONTEXT_RETRIEVAL_TEMPLATE),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptTemplates.CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retrievers[doc], contextualize_prompt
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, custom_rag_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        response = rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=question), response["answer"]])
        return response["answer"]


pdfrag = PdfRAG()
