import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = "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"
with st.sidebar:
    st.title("ChattyPDF")
    st.markdown('''
                About:
                An LLM powered chatbot built using:
                - Streamlit
                - LangChain
                - OpenAI''')
    add_vertical_space(5)
    st.write("Made by [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)")

load_dotenv()
def main():
    st.header("Learn your PDF without reading all of it")

    pdf=st.file_uploader("Upload your PDF",type="pdf")

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                Vectorstore = pickle.load(f)
            # st.write("Embeddings Loaded from Disk")
        else:
            embeddings=OpenAIEmbeddings()
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(Vectorstore,f)
            # st.write("Embeddings Computation Completed")

        memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True,output_key='answer')
        llm=ChatOpenAI(temperature=0.0)
        conversation=ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    memory=memory,
                    retriever=Vectorstore.as_retriever()
                )
        query=st.text_input("Ask question about your file: ")

        if "messages" not in st.session_state:
            st.session_state.messages=[]

        if query:
            st.session_state.messages.append({'role':"user","content": query})

            # st.session_state.convo=conversation
            res=conversation({'question':query})
            st.write(res["answer"])
            st.session_state.messages.append({'role':"assisstant",'content':res["answer"]})

        #     res=st.session_state.convo({'question':query})
        #     st.session_state.chat_history = res['chat_history']
        #     for i,message in enumerate(st.session_state.chat_history):
        #         st.write(message.content)

        # st.session_state.convo

if __name__=='__main__':
    main()