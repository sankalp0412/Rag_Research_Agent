import time
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from neo4j.exceptions import ServiceUnavailable
from langchain_core.prompts.prompt import PromptTemplate

load_dotenv()

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")


def setup_mistral():
    try:
        llm = ChatMistralAI(model="mistral-medium")
        return llm
    except Exception as e:
        print(f"Error setting up Mistral client: {str(e)}")
        return None


def setup_hf():
    try:
        qa_llm = llm = HuggingFaceEndpoint(
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
        return f"Error setting up HF llm:{str(e)} \n"


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


def process_prompt(prompt: str) -> str:
    cypher_llm = setup_mistral()
    if not cypher_llm:
        return "Error: Could not initialize the chatbot"

    qa_llm = setup_hf()
    if not qa_llm:
        return "Error: Could not initialize chatbot"

    kg = setup_kg()
    if not kg:
        return "Error: Could not retrieve information"

    kg.refresh_schema()
    schema = kg.schema

    CYPHER_GENERATION_PROMPT = setup_cypher_prompt()

    chain = GraphCypherQAChain.from_llm(
        cypher_llm,
        qa_llm=qa_llm,
        graph=kg,
        verbose=True,
        allow_dangerous_requests=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )

    try:
        answer = chain.run(prompt)
        return answer
    except Exception as e:
        print(f"Error during chain execution: {str(e)}")
        return "An error occurred while processing the prompt."


def setup_cypher_prompt():
    CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
    You are an AI assistant that generates Cypher queries for a Neo4j knowledge graph.
    The database schema consists of the following nodes and relationships:

     Schema:
     {schema}
     Given a user query, generate a valid and optimized Cypher query that strictly follows this schema.

     **Rules:**

     - Ensure the query is efficient and only returns necessary fields.
     - If no direct match is possible, return the `summary` property of `Paper`.
     - The output must contain only the Cypher query—no explanations or extra text.
     - If the question cant be converted to a valid query reply with "Invalid", without any explanations.

    Examples: Here are a few examples of generated Cypher statements for particular questions:
    # How many papers does the paper "Attention is All you Need" cites 
    MATCH (m:Paper {{name:"Attention is All you Need"}})<-[:CITES]-()
    RETURN count(*) AS citationCount

    The question is:
    {question}"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    return CYPHER_GENERATION_PROMPT


if __name__ == "__main__":
    result = process_prompt("Which paper has the most citations")
    print(result)
    # llm = setup_hf()
    # print(llm.invoke("What is Deep Learning?"))
