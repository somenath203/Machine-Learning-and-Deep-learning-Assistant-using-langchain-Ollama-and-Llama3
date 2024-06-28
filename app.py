from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a helpful machine learning and deep learnig assistant. Please respond to the user's queries. Don't answer anything which is not related to machine learning or deep learning."),
        HumanMessagePromptTemplate.from_template("Question: {question_of_user}")
    ]
)


st.title("A.I. Chat Assistant Powered by Ollama and Llama3")

input_text = st.text_input("Search for the topic you want to know")


llm_model = Ollama(model="llama3")


output_parser = StrOutputParser()


chain = prompt | llm_model | output_parser

if input_text:
    response = chain.invoke({'question_of_user': input_text})
    st.write(response)
