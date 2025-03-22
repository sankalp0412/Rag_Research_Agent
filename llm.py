import time
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from neo4j.exceptions import ServiceUnavailable
from langchain_core.prompts.prompt import PromptTemplate
import streamlit as st

load_dotenv()

NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
HF_API_KEY = st.secrets["HF_API_KEY"]


def setup_mistral():
    try:
        llm = ChatMistralAI(
            model="mistral-medium", mistral_api_key=MISTRAL_API_KEY, temperature=0
        )
        return llm
    except Exception as e:
        print(f"Error setting up Mistral client: {str(e)}")
        return None


def setup_hf():
    try:
        qa_llm = HuggingFaceEndpoint(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            huggingfacehub_api_token=HF_API_KEY,
        )
        return qa_llm
    except Exception as e:
        print(f"Error setting up HF llm:{str(e)} \n")
        return None


def setup_kg():
    try:
        kg = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        return kg
    except ServiceUnavailable:
        print(
            "Error: Could not connect to Neo4j database. Please check if the database is running and accessible."
        )
        return None
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
        return None


def process_prompt(user_query: str) -> str:
    cypher_llm = setup_mistral()
    qa_llm = setup_hf()
    kg = setup_kg()

    if not cypher_llm or not qa_llm or not kg:
        return "Error: Could not initialize required components."

    kg.refresh_schema()
    schema = kg.schema

    chain = GraphCypherQAChain.from_llm(
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        graph=kg,
        verbose=True,
        allow_dangerous_requests=True,
        validate_cypher=True,
    )

    try:
        answer = chain.invoke({"query": user_query})
        if not answer:
            return "No relevant data found."
        return answer.get("result", "No result found.")
    except Exception as e:
        print(f"Error during query execution: {str(e)}")
        return "An error occurred while processing the request."


if __name__ == "__main__":
    result = process_prompt("What is Deep Learning?")
    print(f" Final result: {result}")
