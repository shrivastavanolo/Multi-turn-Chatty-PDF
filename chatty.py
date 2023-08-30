import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


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

        retriever = Vectorstore.as_retriever(search_kwargs=dict(k=1))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        llm = OpenAI(temperature=0) # Can be any valid LLM
        _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
        
        Relevant pieces of previous conversation:
        {history}
        
        (You do not need to use these pieces of information if not relevant)
        
        Current conversation:
        Human: {input}
        AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
        )
        conversation_with_summary = ConversationChain(
            llm=llm,
            prompt=PROMPT,
            # We set a very low max_token_limit for the purposes of testing.
            memory=memory,
            verbose=True
        )
        
        query=st.text_input("Ask question about your file: ")

        if query:
            res=conversation_with_summary.predict(input=query)
            st.write(res)
if __name__=='__main__':
    main()
