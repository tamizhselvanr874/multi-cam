import os
import time
import cv2
import numpy as np
import base64
from PIL import Image
import streamlit as st
from azure.storage.blob import BlobServiceClient, BlobClient
from openai import AzureOpenAI
from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse
import json
import requests
import re

# Azure Configuration
AZURE_STORAGE_SAS_URL = st.secrets["azure"]["storage_sas_url"]
AZURE_OPENAI_ENDPOINT = st.secrets["openai"]["endpoint"]
AZURE_OPENAI_KEY = st.secrets["openai"]["key"]
OPENAI_MODEL = st.secrets["openai"]["model"]
API_VERSION = st.secrets["openai"]["api_version"]

# Parse the SAS URL to get the account URL and SAS token
parsed_url = urlparse(AZURE_STORAGE_SAS_URL)
account_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
sas_token = parsed_url.query

# Initialize Azure clients with SAS token
blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
container_client = blob_service_client.get_container_client("screencapture")

openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Session state management
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'analysis_log' not in st.session_state:
    st.session_state.analysis_log = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Helper Functions
def capture_frame():
    """Capture frame from webcam and upload to Azure"""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert to PNG
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        img_bytes = BytesIO()
        pil_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Upload with timestamp filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        blob_name = f"capture_{timestamp}.png"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(img_bytes, overwrite=True)
        
        return blob_name
    return None

def get_image_base64(image_data):
    """Convert image data to base64 string"""
    return base64.b64encode(image_data).decode('utf-8')

def analyze_image_with_vision(image_data):
    """Analyze image using Azure OpenAI vision capabilities with direct API call"""
    try:
        # Get base64 encoded image
        base64_image = get_image_base64(image_data)
        
        # Define the enhanced system prompt
        system_prompt = """You are a professional security analyst reviewing surveillance footage. 
        For each image, provide:
        1. DETAILED SUMMARY: Describe all important elements including:
           - People (count, approximate age/gender, notable features)
           - Objects (weapons, suspicious packages, etc.)
           - Activities (normal behavior vs suspicious actions)
           - Environment (location type, lighting, weather if visible)
        2. THREAT ASSESSMENT: One of these levels:
           - [SAFE]: Normal activity, no threats detected
           - [LOW RISK]: Minor concerns but likely harmless
           - [MEDIUM RISK]: Suspicious activity that warrants attention
           - [HIGH RISK]: Clear danger requiring immediate action
        3. REASONING: Detailed explanation for the threat assessment
        4. RECOMMENDATION: Suggested response if threat is detected
        
        Format your response EXACTLY as:
        SUMMARY: [detailed description] 
        THREAT: [SAFE/LOW RISK/MEDIUM RISK/HIGH RISK]
        REASON: [detailed reasoning]
        RECOMMENDATION: [action suggestion]"""

        # Prepare the API request payload with image
        payload = {
            "model": OPENAI_MODEL,
            "max_tokens": 1000,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this surveillance image for security threats:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }

        # Make the direct API call
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_KEY
        }
        
        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{OPENAI_MODEL}/chat/completions?api-version={API_VERSION}"
        
        # Add a timeout and retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Analyzing image (attempt {attempt+1}/{max_retries})..."):
                    response = requests.post(
                        url, 
                        headers=headers, 
                        json=payload,
                        timeout=30  # 30 second timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'choices' in result and len(result['choices']) > 0:
                            analysis = result['choices'][0]['message']['content']
                            return analysis
                    
                    # If we get here, there was an issue with the response
                    time.sleep(2)  # Wait before retrying
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    return f"Analysis failed after {max_retries} attempts: {str(e)}"
        
        return "Failed to get a valid response from the analysis service."
    except Exception as e:
        return f"Analysis failed: {str(e)}"

def fetch_latest_images():
    """Fetch all images from Azure Storage"""
    blobs = container_client.list_blobs()
    images = [blob.name for blob in blobs if blob.name.endswith('.png')]
    # Sort images by timestamp in filename
    images.sort()
    return images

def parse_analysis_result(result):
    """Parse the enhanced analysis result into components using regex for more reliability"""
    sections = {
        "SUMMARY": "Not available",
        "THREAT": "Not available",
        "REASON": "Not available",
        "RECOMMENDATION": "None"
    }
    
    # Use regex to find each section
    summary_match = re.search(r'SUMMARY:(.+?)(?=THREAT:|$)', result, re.DOTALL)
    if summary_match:
        sections["SUMMARY"] = summary_match.group(1).strip()
    
    threat_match = re.search(r'THREAT:(.+?)(?=REASON:|$)', result, re.DOTALL)
    if threat_match:
        threat_text = threat_match.group(1).strip()
        # Normalize the threat level
        if "SAFE" in threat_text:
            sections["THREAT"] = "SAFE"
        elif "LOW RISK" in threat_text:
            sections["THREAT"] = "LOW RISK"
        elif "MEDIUM RISK" in threat_text:
            sections["THREAT"] = "MEDIUM RISK"
        elif "HIGH RISK" in threat_text:
            sections["THREAT"] = "HIGH RISK"
        else:
            sections["THREAT"] = threat_text
    
    reason_match = re.search(r'REASON:(.+?)(?=RECOMMENDATION:|$)', result, re.DOTALL)
    if reason_match:
        sections["REASON"] = reason_match.group(1).strip()
    
    recommendation_match = re.search(r'RECOMMENDATION:(.+?)$', result, re.DOTALL)
    if recommendation_match:
        sections["RECOMMENDATION"] = recommendation_match.group(1).strip()
    
    return sections["SUMMARY"], sections["THREAT"], sections["REASON"], sections["RECOMMENDATION"]

# App Layout
st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["Capture App", "Admin Dashboard"])

with tab1:
    st.header("📸 Capture App")
    st.write("Automatically captures and uploads images every 5 seconds")
    
    if st.button("Start Capture"):
        placeholder = st.empty()
        stop_button = st.button("Stop Capture")
        
        while not stop_button:
            with placeholder.container():
                blob_name = capture_frame()
                if blob_name:
                    st.success(f"Uploaded: {blob_name}")
                else:
                    st.error("Failed to capture image")
            
            time.sleep(5)
            stop_button = st.button("Stop Capture")

with tab2:
    st.header("🧑‍💼 Admin Dashboard")
    st.write("Real-time monitoring and threat analysis")
    
    # Auto-refresh logic
    refresh_interval = 10  # seconds
    current_time = time.time()
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.captured_images = fetch_latest_images()
        st.session_state.last_refresh = current_time
    
    # Manual refresh button
    if st.button("Refresh Images Now"):
        st.session_state.captured_images = fetch_latest_images()
        st.session_state.last_refresh = time.time()
    
    if not st.session_state.captured_images:
        st.session_state.captured_images = fetch_latest_images()

    # Set default index to the latest image if we have images
    if st.session_state.captured_images and st.session_state.current_index >= len(st.session_state.captured_images):
        st.session_state.current_index = len(st.session_state.captured_images) - 1
    
    if st.session_state.captured_images:
        # Show total images count
        st.write(f"Total images: {len(st.session_state.captured_images)}")
        
        # Navigation controls
        col1, col2, col3, col4 = st.columns([1,1,1,2])
        with col1:
            if st.button("⏮️ First") and len(st.session_state.captured_images) > 0:
                st.session_state.current_index = 0
        with col2:
            if st.button("⏪ Previous") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
        with col3:
            if st.button("Next ⏩") and st.session_state.current_index < len(st.session_state.captured_images) - 1:
                st.session_state.current_index += 1
        with col4:
            if st.button("Latest ⏭️") and len(st.session_state.captured_images) > 0:
                st.session_state.current_index = len(st.session_state.captured_images) - 1
        
        # Display current image index
        st.write(f"Viewing image {st.session_state.current_index + 1} of {len(st.session_state.captured_images)}")
        
        current_blob = st.session_state.captured_images[st.session_state.current_index]
        blob_client = container_client.get_blob_client(current_blob)
        image_data = blob_client.download_blob().readall()
        
        # Display image and analysis in columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_data, caption=current_blob, use_column_width=True)
        
        with col2:
            st.subheader("Image Analysis")
            
            # Check if analysis exists in cache
            analysis_key = f"analysis_{current_blob}"
            
            if analysis_key not in st.session_state.analysis_cache:
                with st.spinner("Analyzing image... (This may take a few seconds)"):
                    # Perform new analysis with the direct vision API approach
                    analysis_result = analyze_image_with_vision(image_data)
                    
                    # Cache the result
                    st.session_state.analysis_cache[analysis_key] = analysis_result
                    
                    # Parse and log this analysis
                    summary, threat, reason, recommendation = parse_analysis_result(analysis_result)
                    log_entry = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": current_blob,
                        "summary": summary,
                        "threat": threat,
                        "reason": reason,
                        "recommendation": recommendation,
                        "raw_analysis": analysis_result
                    }
                    
                    # Add to log if not already present
                    image_exists = any(entry['image'] == current_blob for entry in st.session_state.analysis_log)
                    if not image_exists:
                        st.session_state.analysis_log.append(log_entry)
            
            # Display analysis
            analysis_result = st.session_state.analysis_cache.get(analysis_key, "Analysis not available")
            summary, threat, reason, recommendation = parse_analysis_result(analysis_result)
            
            # Display with better formatting
            st.markdown("### Summary")
            st.write(summary)
            
            # Color-code threat level
            threat_color = {
                "SAFE": "green",
                "LOW RISK": "blue",
                "MEDIUM RISK": "orange",
                "HIGH RISK": "red"
            }.get(threat, "gray")
            
            st.markdown(f"### Threat Level: :{threat_color}[{threat}]")
            
            st.markdown("### Reason")
            st.write(reason)
            
            st.markdown("### Recommendation")
            st.write(recommendation)
            
            # Option to reanalyze if needed
            if st.button("Reanalyze This Image"):
                with st.spinner("Reanalyzing image..."):
                    analysis_result = analyze_image_with_vision(image_data)
                    st.session_state.analysis_cache[analysis_key] = analysis_result
                    
                    # Update log
                    summary, threat, reason, recommendation = parse_analysis_result(analysis_result)
                    for i, entry in enumerate(st.session_state.analysis_log):
                        if entry['image'] == current_blob:
                            st.session_state.analysis_log[i] = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (Reanalyzed)",
                                "image": current_blob,
                                "summary": summary,
                                "threat": threat,
                                "reason": reason,
                                "recommendation": recommendation,
                                "raw_analysis": analysis_result
                            }
                    st.experimental_rerun()
        
        # Display log
        st.subheader("Security Insights Log")
        
        # Display filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_threat = st.selectbox(
                "Filter by threat level:",
                ["All", "SAFE", "LOW RISK", "MEDIUM RISK", "HIGH RISK"]
            )
        
        # Apply filters
        filtered_log = st.session_state.analysis_log.copy()
        if filter_threat != "All":
            filtered_log = [entry for entry in filtered_log if entry["threat"] == filter_threat]
        
        # Sort by timestamp (newest first)
        filtered_log.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Display log entries
        if filtered_log:
            for entry in filtered_log:
                threat_color = {
                    "SAFE": "green",
                    "LOW RISK": "blue",
                    "MEDIUM RISK": "orange",
                    "HIGH RISK": "red"
                }.get(entry["threat"], "gray")
                
                with st.expander(f"{entry['timestamp']} - {entry['image']} - :{threat_color}[{entry['threat']}]"):
                    st.write(f"**Summary:** {entry['summary']}")
                    st.write(f"**Reason:** {entry['reason']}")
                    st.write(f"**Recommendation:** {entry['recommendation']}")
                    
                    # Add button to view this image
                    image_index = st.session_state.captured_images.index(entry['image']) if entry['image'] in st.session_state.captured_images else -1
                    if image_index >= 0 and st.button(f"View this image", key=f"btn_{entry['image']}"):
                        st.session_state.current_index = image_index
                        st.experimental_rerun()
        else:
            st.info("No log entries match the selected filters")
    else:
        st.warning("No images found in storage")

# Add automatic refresh by rerunning the script periodically
if time.time() - st.session_state.last_refresh > refresh_interval:
    st.session_state.last_refresh = time.time()
    st.experimental_rerun()
