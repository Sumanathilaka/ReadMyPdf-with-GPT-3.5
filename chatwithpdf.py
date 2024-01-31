
import streamlit as st
import os, yaml, textwrap, urljoin
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from htmlTemplates import css, bot_template, user_template


with open('credentials.yaml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

os.environ['OPENAI_API_KEY'] = credentials['OPENAI_API_KEY']

chat_llm = ChatOpenAI(
                openai_api_key = os.environ['OPENAI_API_KEY'],
                model = 'gpt-3.5-turbo',
                temperature=0.5,
                max_tokens=500
                )
embedding_llm = HuggingFaceBgeEmbeddings(
                                        model_name = "BAAI/bge-small-en",
                                        model_kwargs = {'device': 'cuda'},
                                        encode_kwargs = {'normalize_embeddings': False}
                                        )


DATA_PATH='data/'

#cleaning the repository
def clean_directory(directory_path):
    try:
        if os.path.exists(directory_path):
            # Iterate over all files in the directory and remove them
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Directory '{directory_path}' cleaned successfully.")
        else:
            print(f"Directory '{directory_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    
#Reading the Scraped pdf from the directory
def get_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            text += page.extract_text()
    return text

#Creating the tex Chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
                                        separator="\n",
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        )
    chunks = text_splitter.split_text(text)
    return chunks

#buidling the vector store
def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(
                                    texts=text_chunks, 
                                    embedding=embedding_llm
                                    )
    return vectorstore

#Pipeline
def data_pipeline(url):
    text = get_pdf_text(url)
    chunks = get_text_chunks(text)
    vectorstore = get_vectorstore(chunks)
    return vectorstore 

#Conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
                                    memory_key='chat_history', 
                                    return_messages=True
                                    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                                            llm=chat_llm,
                                                            retriever=vectorstore.as_retriever(),
                                                            memory=memory
                                                            )
    return conversation_chain



#user input handling and chat 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    
    
    #print(j)
    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:            
            
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
                       
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

#main method
        
def main():
    
    st.set_page_config(page_title="Chat with Pdf:",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
     

    st.header("Chat with PDFBOT :robot_face:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if st.button("Ask"):
        with st.spinner("Searching for the best answer"):
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your document location")
        web_url = st.text_input("Enter the Website  URL")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # scrape the pdf text
                vectorstore=data_pipeline(web_url)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
       
        

if __name__ == '__main__':
    main()
