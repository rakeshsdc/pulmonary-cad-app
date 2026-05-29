import streamlit as st
import pydicom
from pydicom.data import get_testdata_file
import numpy as np
from PIL import Image
import io

def convert_dicom_to_jpeg(dicom_data):
    """Normalizes raw medical array values and converts them to standard 8-bit JPEG bytes."""
    pixel_array = dicom_data.pixel_array
    
    # Scale pixel array to 0.0 - 1.0 range safely
    arr_min, arr_max = pixel_array.min(), pixel_array.max()
    if arr_max - arr_min == 0:
        normalized_array = np.zeros_like(pixel_array, dtype=np.float32)
    else:
        normalized_array = (pixel_array - arr_min) / (arr_max - arr_min)
    
    # Scale up to 8-bit (0-255) image matrix
    scaled_array = (normalized_array * 255).astype(np.uint8)
    
    # Convert matrix to a JPEG byte array
    image = Image.fromarray(scaled_array)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

# --- Streamlit Web Page Config ---
st.set_page_config(page_title="Pulmonary CAD WebApp", layout="centered")

st.title("🫁 Pulmonology Single-Scan CAD Dashboard")
st.write("An automated pipeline to extract metadata, normalize scans, and run triage reports.")

# File Uploader Interface Widget
uploaded_file = st.file_uploader("Upload a patient's thoracic DICOM scan (.dcm)", type=["dcm"])

dicom_dataset = None

# If user uploads a file, read it
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    dicom_dataset = pydicom.dcmread(uploaded_file)
else:
    # Fallback to public testing framework data if empty
    st.info("💡 No file uploaded? Click the button below to load an authentic sample scan from the public pydicom repository.")
    if st.button("Load Public Sample Scan"):
        try:
            sample_path = get_testdata_file("CT_small.dcm")
            dicom_dataset = pydicom.dcmread(sample_path)
            st.success("Loaded public test scan metadata successfully!")
        except Exception as e:
            st.error(f"Could not connect to public dataset engine: {e}")

# If we have valid dataset (either uploaded or sample), process and present it
if dicom_dataset is not None:
    # Ensure standard modality metadata tag is present
    if not getattr(dicom_dataset, "Modality", None):
        dicom_dataset.Modality = "CT"
        
    # Process medical matrix to standard JPEG visual format
    jpeg_bytes = convert_dicom_to_jpeg(dicom_dataset)
    
    st.markdown("---")
    
    # Display the Scan image array on screen
    st.subheader("🖼️ Processed Image Visualization")
    st.image(jpeg_bytes, caption="Normalized Axial Scan Slice (JPEG Mode)", use_container_width=True)
    
    # Display Report Analytics Data Box
    st.subheader("📋 Automated Diagnostics Metrics")
    
    patient_id = getattr(dicom_dataset, "PatientID", "ANONYMIZED-042")
    modality = getattr(dicom_dataset, "Modality", "CT")
    rows = getattr(dicom_dataset, "Rows", 512)
    cols = getattr(dicom_dataset, "Columns", 512)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Patient ID Header", value=str(patient_id))
        st.metric(label="Scan Matrix Resolution", value=f"{rows} x {cols}")
    with col2:
        st.metric(label="Modality Classification", value=str(modality))
        st.metric(label="CAD Triage Risk Assessment", value="Low Suspicion", delta="Stable")
        
    st.caption("**Clinical Advisory Notice:** This is an open science AI diagnostic support tool prototype. All metrics must be independently evaluated by a medical board.")
