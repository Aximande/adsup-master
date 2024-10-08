# File: pages/2_Generate.py

import streamlit as st
import replicate
import os
import csv
import uuid
import json
from datetime import datetime
from transformers import pipeline

# Function to retrieve the API token
def get_replicate_api_token():
    # Attempt to retrieve the API token from Streamlit secrets
    token = st.secrets.get("REPLICATE_API_TOKEN")
    if not token:
        # Fallback to environment variable
        token = os.getenv("REPLICATE_API_TOKEN")
    return token

# Retrieve the API token
REPLICATE_API_TOKEN = get_replicate_api_token()

# Check if the API token was successfully retrieved
if not REPLICATE_API_TOKEN:
    st.error("Replicate API token not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

# Initialize Replicate client
try:
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
except Exception as e:
    st.error(f"Failed to initialize Replicate client: {e}")
    st.stop()

# Function to test authentication by fetching account info
def test_authentication():
    try:
        account = replicate_client.account()  # Fetch authenticated account information
        st.success(f"Authenticated successfully!\nAccount Type: {account.type.capitalize()}, Username: {account.username}")
        return True
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Authentication failed: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during authentication: {e}")
        return False

# Execute the authentication test
if not test_authentication():
    st.stop()

# Rest of your code...
