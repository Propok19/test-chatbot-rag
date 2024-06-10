import time
import streamlit as st
from services.rag import pdfrag


class AnnualReportChatbot:
    def __init__(self):
        self.doc_name = None
        self.messages = st.session_state.get("messages", [])

    def get_doc_name(self):
        return (
            "documents/Airbus-Annual-Report-2023.pdf"
            if st.session_state["doc_name"] == "Airbus Annual Report 2023"
            else "documents/annualreport2223.pdf"
        )

    @staticmethod
    def stream_data(text):
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.02)

    def process_messages(self):
        for message in self.messages:
            st.chat_message("human").write(message[0])
            st.chat_message("ai").write(message[1])

        if query := st.chat_input():
            st.chat_message("human").write(query)
            response = pdfrag.get_answer(query, self.get_doc_name())
            st.chat_message("ai").write_stream(self.stream_data(response))
            self.messages.append([query, response])
            st.session_state["messages"] = self.messages

    def run(self):
        st.set_page_config(page_title="RAG ChatBot")
        st.title("Annual Report Chatbot")
        st.markdown("This chatbot aims to answer user questions based on annual reports of two companies: Airbus and Singapore Airlines. Chatbot uses the RAG technique to fetch relevant contexts from documents to answer questions. ")
        st.markdown("Choose a document to query:")
        self.doc_name = st.selectbox(
            "Choose a document to query:",
            ("Airbus Annual Report 2023", "Singapore Airlines Annual Report 2022-23"),
            key="doc_name",
            label_visibility="collapsed"
        )
        self.process_messages()


if __name__ == "__main__":
    chatbot = AnnualReportChatbot()
    chatbot.run()
