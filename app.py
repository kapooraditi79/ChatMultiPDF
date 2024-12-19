import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_read=PdfReader(pdf)
        for page in pdf_read.pages:
            text+=page.extract_text()
    return text

def get_pdf_chunks(pdf_text):
    splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len #lambda x: len(x), where x is pdf_text[each_chunk]
    )
    pdf_text_chunk=splitter.split_text(pdf_text)
    return pdf_text_chunk


def form_vectorstore(pdf_chunks):
    try:
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
               
        vectorstore=FAISS.from_texts(texts=pdf_chunks,embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error: {e}")
        return [{e}]


def get_conversation_chain(vectorstore):

    llm=ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=100,
    )
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_message=True,
    )
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    try:

        response=st.session_state.conversation({
            'question':user_question,
            'chat_history': st.session_state.chat_history or []
            })
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    except Exception as e:
        st.write(f"Error: {e}")
        
#os.environ["HUGGINGFACE_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")

def main():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

    st.set_page_config(page_title="Upload and Chat", page_icon="chat with book")
    st.header("Upload and Chat")    

    st.write(css, unsafe_allow_html=True)
   
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    

    user_question=st.text_input("Ask Question about uploads")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Uploads")
        pdf_docs=st.file_uploader("Upload your pdf and click Process", accept_multiple_files=True)
        

    if st.button("Process"):
        with st.spinner("processing..."):
            pdf_text=get_pdf_text(pdf_docs)

            pdf_chunks=get_pdf_chunks(pdf_text)
            
            pdf_vectorstore=form_vectorstore(pdf_chunks)

            st.session_state.conversation=get_conversation_chain(pdf_vectorstore)
            


if __name__ == "__main__":
    main()