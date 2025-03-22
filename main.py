import streamlit as st # type: ignore
import pandas as pd
from dify import upload_papers
from llm import process_prompt
import time
import requests
from requests.exceptions import RequestException

col1, col2 = st.columns([3, 1])  # Adjust column sizes
mistral_url = "https://mistral.ai/"
with col1:
    st.header("RAG Based Research Chatbot powered by :red[Mistral]" )
with col2:
    # Make sure the image path is correct; adjust the relative path as needed
    st.image("utils/mistral_logo.png", use_container_width=True)  # Adjust for your image path
    st.markdown(f"[Visit Mistral AI]({mistral_url})", unsafe_allow_html=True)

input_papers = st.file_uploader(label="Add Your Research Papers", type="pdf", accept_multiple_files=True)

# Variable to track if upload was successful
upload_success = False

if len(input_papers):
    with st.spinner("Wait for it...", show_time=True):
        messages = upload_papers(input_papers, False)
    for status, summary, metadata, kg_overview in messages: 
        if 'Failed' in status:
            st.error(status, icon='🚨')
        else:
            st.success(status, icon='✅')
            st.markdown("### Information Regarding Your Paper")
            
            # Clean up summary and metadata
            clean_summary = summary.strip().strip('"""').strip()
            clean_metadata = metadata.strip().strip('"""').strip().replace('"Here is the metadata of the Paper":\n', '')
            
            # Split metadata into lines for better formatting
            metadata_lines = clean_metadata.split('\n')
            if metadata_lines:
                metadata_lines.pop(0)
            
            # Display the information
            st.markdown(f"""
            :orange[Summary:]  
            {clean_summary} """)
            
     
            st.markdown("""
            :orange[Metadata:]  
            """)
            
            # Display metadata as a list
            for line in metadata_lines:
                st.markdown(f"- {line}")

            upload_success = True  # Set to True only if success

# Show chat box only if upload was successful
if upload_success:
    st.subheader(":red[Chat with an Agent regarding your Research Papers]")
    prompt = st.chat_input("Ask a question about your papers:")
    if prompt:
        message = st.chat_message("user")
        message.write(prompt)
        #Get answer from the llm using kowledge graph
        with st.spinner("Looking for answer...", show_time=True):
            answer = process_prompt(prompt)
        reply = st.chat_message("assistant")
        reply.write(answer)
        