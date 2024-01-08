import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
from langchain_community.chat_models import openai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template 


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs: 
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()                # estrae le file di testo dal pdf
    return text

def get_text_chunks(text): 
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") - Huggingface funziona in locale, OpenAI in cloud a pagamento 
    vectorstore =faiss.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = openai()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory
     )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) #Contiene tutta la configurazione da vectorestore e memory, quindi ricorda tutto il contesto precedente
    st.session_state.chat_history = response({'chat_history'})

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    



def main():
    load_dotenv()
    st.set_page_config(page_title="📙Controlla i tuoi file personali📘")

    st.write(css, unsafe_allow_html=True)

    if "conversation"  not in st.session_state:
        st.session_state.conversation = None
    if "chat_hostory" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Controlla i tuoi file! 📚📖")
    user_question = st.text_input("Vorrei vedere se nei miei file c'è")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("i tuoi documenti")
        pdf_docs = st.file_uploader(
            "Carica i documenti e clicca su procedi", accept_multiple_files=True)
        if st.button("Procedi"):
            with st.spinner("Caricamento"):
                # testo del pdf 
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
            
                # servono parti singole del pdf
                text_chunks = get_text_chunks(raw_text)

                # vector store degli embeds
                vectorstore = get_vectorstore(text_chunks)

                #creare conversazione
                st.session_state.conversation = get_conversation_chain(vectorstore)
    


if __name__ == '__main__':
    main()


