from typing import IO, List, Tuple
import requests
import os
import json
from dotenv import load_dotenv
import time
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
DIFY_API_KEY = st.secrets["DIFY_API_KEY"]
if not DIFY_API_KEY:
    raise ValueError("DIFY_API_KEY is not set in your environment.")

BASE_URL = "https://api.dify.ai/v1"


@st.cache_data(show_spinner=False)
def upload_papers(
    files: List[IO], dummy_response: bool
) -> List[Tuple[str, str | None, str | None, str | None]]:
    """Uploads all papers to the Dify API and processes them."""
    results = []
    for file in files:
        print(f"Name: {file.name}")
        result = use_dify(file)
        results.append(result)
    return results


@st.cache_data(show_spinner=False)
def use_dify(file: IO) -> Tuple[str, str | None, str | None, str | None]:
    """Uploads a file to Dify and runs the workflow."""
    print(f"Processing file: {file.name}")
    file_id = file_upload_dify(file)
    if file_id:
        print(f"Uploaded {os.path.basename(file.name)} to Dify with file ID: {file_id}")
        status, summary, metadata, kg_overview = run_dify_workflow(file_id)
        if status == "succeeded":
            print(f"Summary: {summary}")
            print(f"Metadata: {metadata}")
            print(f"Knowledge Graph Overview: {kg_overview}")
            return (
                f"Success: {os.path.basename(file.name)} added to knowledge graph",
                summary,
                metadata,
                kg_overview,
            )
        else:
            return (
                f"Failed to add {os.path.basename(file.name)} to knowledge graph",
                None,
                None,
                None,
            )
    else:
        print(f"Failed to upload file to Dify: {os.path.basename(file.name)}")
        return (f"Upload failed for {os.path.basename(file.name)}", None, None, None)


@st.cache_data(show_spinner=False)
def run_dify_workflow(file_id: str) -> Tuple[str, str | None, str | None, str | None]:
    """Runs a Dify workflow with the given file ID."""
    workflow_run_url = f"{BASE_URL}/workflows/run"

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = json.dumps(
        {
            "inputs": {
                "research_paper": {
                    "type": "document",
                    "transfer_method": "local_file",
                    "upload_file_id": file_id,
                }
            },
            "response_mode": "blocking",
            "user": "abc-123",
        }
    )

    try:
        response = requests.post(workflow_run_url, headers=headers, data=data)
        if response.status_code in (200, 201):
            res_json = response.json()
            print(f"Workflow response: {res_json}")
            status = res_json.get("data", {}).get("status")
            if status == "failed":
                print(
                    f"Upload to knowledge graph failed due to: {res_json['data']['error']}"
                )
                return "failed", None, None, None
            return handle_output(res_json)
        else:
            print(
                f"Error calling workflow API: {response.status_code} - {response.text}"
            )
            return "failed", None, None, None
    except requests.exceptions.RequestException as e:
        print(f"Error during workflow API request: {e}")
        return "failed", None, None, None


@st.cache_data(show_spinner=False)
def file_upload_dify(file: IO) -> str | None:
    """Handles the actual file upload to the Dify API."""
    file_upload_url = f"{BASE_URL}/files/upload"

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
    }
    files = {
        "file": (os.path.basename(file.name), file, "application/pdf"),
    }
    data = {"user": "abc-123"}

    try:
        response = requests.post(
            file_upload_url, headers=headers, files=files, data=data
        )
        if response.status_code in (200, 201):
            response_data = response.json()
            return response_data.get("id")
        else:
            print(f"Upload failed with status {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during file upload API request: {e}")
        return None


def handle_output(
    workflow_response: dict,
) -> Tuple[str, str | None, str | None, str | None]:
    """Extracts summary, metadata, and knowledge graph overview from workflow response."""
    outputs = workflow_response.get("data", {}).get("outputs", {})
    summary = outputs.get("Summary")
    metadata = outputs.get("metadata")
    kg_overview = outputs.get("knowledge_graph_overview")
    status = workflow_response.get("data", {}).get("status", "failed")
    return status, summary, metadata, kg_overview


if __name__ == "__main__":
    # Example with file upload
    file_path = "../Input_papers/Attention.pdf"
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    with open(file_path, "rb") as f:
        results = upload_papers([f], False)
        for result in results:
            status, summary, metadata, kg_overview = result
            print(f"Result: {status}")
            if summary:
                print(f"Summary: {summary}")
            if metadata:
                print(f"Metadata: {metadata}")
            if kg_overview:
                print(f"Knowledge Graph Overview: {kg_overview}")
