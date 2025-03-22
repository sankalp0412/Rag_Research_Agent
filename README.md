# RAG Based AI Academic Assistant Chatbot

## About
A specialized chatbot that helps researchers by understanding and answering questions about academic papers. It uses knowledge graphs to store paper relationships and multiple AI models to generate accurate responses.

## Features
- **Smart Paper Understanding**: Processes and understands academic papers using RAG.
- **Knowledge Graph Storage**: Maintains paper relationships in Neo4j database>
- **Multi-Model Pipeline**: Uses Mistral AI and Hugging Face models.
- **Research Integration**: Connects with Semantic Scholar for paper metadata.
- **Conversation Management**: Uses DIFY workflow for Extracting Raw data from the research paper, and adding it to the Knowledge Graph.

## Technologies Used
- Python 3.9+
- Streamlit for UI
- Neo4j Graph Database
- LangChain Framework
- Mistral AI & Hugging Face
- Semantic Scholar API
- DIFY

## My Learnings
- Building complex AI pipelines using LangChain.
- Working with knowledge graphs in Neo4j.
- Integrating multiple AI models and APIs.
- Managing research data through Semantic Scholar.
- Implementing a Workflow using DIFY

## How It Works

The system follows an integrated pipeline for processing research papers and answering user queries:

1. **User Input**: 
   - The user uploads a research paper in PDF format.
   
2. **Dify Workflow**: 
   - Dify extracts raw data from the PDF, including text and metadata.
   - **Mistral AI** is used to summarize the extracted content.
   - The **Semantic Scholar API** retrieves additional paper metadata.
   - The metadata and relevant details are inserted into the **Neo4j knowledge graph**.
   
3. **Answering Queries**:
   - For any user question related to the paper, the system queries the knowledge graph using **Langchain** to perform **Retrieval-Augmented Generation (RAG)**.
   - **Mistral AI** and **Hugging Face models** are used to generate the final response based on the context and knowledge retrieved.

This entire process is seamlessly managed by **Dify.AI**, ensuring smooth workflow and efficient conversation handling.

[Checkout the project :rocket:](https://sankalp0412-rag-research-agent-main-rkebki.streamlit.app/)

## DIFY Worfflow:

![image](https://github.com/user-attachments/assets/d631a070-9f03-45c0-b329-aa4b33f1a2f6)


## UI:

- Add Paper  
  ![Screenshot 1](https://github.com/user-attachments/assets/fcaee54b-4bed-4f5a-83ab-015d66908430)

- Get Summary
  ![Screenshot 2](https://github.com/user-attachments/assets/c230ade4-5780-40ca-8a11-48c11178cd8b)

- Get Metadata 
  ![Screenshot 3](https://github.com/user-attachments/assets/8a17f20d-2e76-4f88-a19b-1fa3c2c95073)

- Cross Question 
  ![Screenshot 4](https://github.com/user-attachments/assets/17f723b6-b18a-4e37-8877-d04f94709a90)

- Cross Question 
![Screenshot 5](https://github.com/user-attachments/assets/7636c30f-849c-4c61-9945-df381e66d7d1)




