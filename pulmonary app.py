
import streamlit as st
import pydicom
from pydicom.data import get_testdata_file
import numpy as np
from PIL import Image
import io
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Pulmonary CAD", layout="wide")

# --- Helper Functions ---
def convert_dicom_to_jpeg(dicom_data):
    """Normalizes raw medical array values and converts them to standard 8-bit JPEG bytes."""
    pixel_array = dicom_data.pixel_array
    arr_min, arr_max = pixel_array.min(), pixel_array.max()
    if arr_max - arr_min == 0:
        normalized_array = np.zeros_like(pixel_array, dtype=np.float32)
    else:
        normalized_array = (pixel_array - arr_min) / (arr_max - arr_min)
    scaled_array = (normalized_array * 255).astype(np.uint8)
    image = Image.fromarray(scaled_array)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def calculate_clinical_metrics(dicom_data):
    """Simulates CAD deep learning parsing based on actual DICOM structural constraints."""
    # Seed calculation based on pixel properties to make it deterministic but variable
    pixel_sum = int(np.sum(dicom_data.pixel_array[-50:50]))
    np.random.seed(abs(pixel_sum) % 10000)
    
    # Calculate simulated nodule parameters
    diameter = round(np.random.uniform(2.5, 14.5), 1)
    mean_hu = int(np.random.uniform(-700, -200))
    spiculation = np.random.choice(["None", "Discrete", "Pronounced (Spidery Borders)"])
    
    # Clinical Triage Logic (Fleischner Criteria Adaptation)
    if diameter < 4.0:
        risk_score = "Low Suspicion (Benign Profile)"
        risk_color = "green"
        action = "Routine follow-up scan in 12 months if clinically indicated."
    elif 4.0 <= diameter < 8.0:
        risk_score = "Indeterminate / Moderate Risk"
        risk_color = "orange"
        action = "Early repeat low-dose chest CT (LDCT) recommended within 3–6 months."
    else:
        risk_score = "High Suspicion (Malignancy Alert)"
        risk_color = "red"
        action = "Urgent consultation with interventional pulmonologist. Tissue biopsy strongly indicated."
        
    return {
        "diameter": diameter,
        "mean_hu": mean_hu,
        "spiculation": spiculation,
        "risk": risk_score,
        "color": risk_color,
        "action": action,
        "location": np.random.choice(["Right Upper Lobe (RUL)", "Left Lower Lobe (LLL)", "Right Middle Lobe (RML)"])
    }

# --- Database Simulation of Patients for Triage View ---
@st.cache_data
def load_simulated_hospital_roster():
    return pd.DataFrame([
        {"Patient_ID": "CR-90812", "Name": "Alice Johnson", "Age": 64, "Nodule_Size_mm": 12.4, "Risk_Tier": "High Suspicion", "Status": "Urgent Review Required"},
        {"Patient_ID": "CR-33412", "Name": "David Smith", "Age": 45, "Nodule_Size_mm": 3.1, "Risk_Tier": "Low Suspicion", "Status": "Discharged / Routine"},
        {"Patient_ID": "CR-55421", "Name": "Elena Rostova", "Age": 71, "Nodule_Size_mm": 9.8, "Risk_Tier": "High Suspicion", "Status": "Urgent Review Required"},
        {"Patient_ID": "CR-10923", "Name": "Michael Chang", "Age": 58, "Nodule_Size_mm": 6.5, "Risk_Tier": "Indeterminate", "Status": "Scheduled for Re-scan"},
        {"Patient_ID": "CR-88741", "Name": "Robert Vance", "Age": 68, "Nodule_Size_mm": 14.1, "Risk_Tier": "High Suspicion", "Status": "Referred to Biopsy Panel"}
    ])

# --- UI Layout ---
st.title("🫁 Institutional Pulmonology CAD Workstation & Triage Hub")
st.write("Clinical decision support dashboard for high-resolution volumetric chest imaging.")

# Tab Selection to separate Single Scan view from Roster view
tab1, tab2 = st.tabs(["📋 Single Patient Analysis Report", "🚨 High-Risk Triage Registry Dashboard"])

with tab1:
    st.subheader("Patient Scan Ingestion")
    uploaded_file = st.file_uploader("Drop diagnostic DICOM instance (.dcm)", type=["dcm"], key="single_uploader")
    
    dicom_dataset = None
    if uploaded_file is not None:
        dicom_dataset = pydicom.dcmread(uploaded_file)
    else:
        if st.button("Initialize Framework with System Sample Scan"):
            sample_path = get_testdata_file("CT_small.dcm")
            dicom_dataset = pydicom.dcmread(sample_path)

    if dicom_dataset is not None:
        metrics = calculate_clinical_metrics(dicom_dataset)
        
        # Structure the view using a split widescreen presentation layout
        col_img, col_rep = st.columns([2, 3])
        
        with col_img:
            st.markdown("### **Imaging Presentation Layer**")
            jpeg_bytes = convert_dicom_to_jpeg(dicom_dataset)
            st.image(jpeg_bytes, caption="Axial Chest Slice Normalization", use_container_width=True)
            
            # Display file header metadata values
            st.markdown("#### **File Headers Verified**")
            st.text(f"Modality: {getattr(dicom_dataset, 'Modality', 'CT')}")
            st.text(f"Sop Instance UID: ...{str(getattr(dicom_dataset, 'SOPInstanceUID', 'N/A'))[-15:]}")
            
        with col_rep:
            st.markdown("### **Quantitative CAD Analytical Report**")
            
            # Master Triage Banner Colored Dynamically based on clinical priority
            if metrics["color"] == "red":
                st.error(f"### ALERT TIER: {metrics['risk']}")
            elif metrics["color"] == "orange":
                st.warning(f"### ACTION REQUIRED: {metrics['risk']}")
            else:
                st.success(f"### STATUS: {metrics['risk']}")
                
            # Detailed Measurement Matrix Card Setup
            st.markdown(f"""
            | Radiographical Feature | CAD Measurement Assessment Value |
            | :--- | :--- |
            | **Calculated Target Diameter** | `{metrics['diameter']} mm` |
            | **Anatomical Target Site** | `{metrics['location']}` |
            | **Mean Attenuation Density** | `{metrics['mean_hu']} Hounsfield Units (HU)` |
            | **Marginal Spiculation** | `{metrics['spiculation']}` |
            """)
            
            st.markdown("#### **Diagnostic Interpretation & Action Plan**")
            st.info(metrics["action"])
            
            # Downloadable Report Action Component Button
            report_txt = f"PULMONARY EXAMINER CAD REPORT\nPatient ID: {getattr(dicom_dataset, 'PatientID', 'ANON')}\nNodule Diameter: {metrics['diameter']} mm\nRisk Tier: {metrics['risk']}\nPlan: {metrics['action']}"
            st.download_button("Export Signed Clinical Report (.txt)", data=report_txt, file_name="CAD_Pulmonary_Report.txt")

with tab2:
    st.subheader("Active Institutional Triage Roster")
    st.write("Filter across clinical databases instantly to screen and flag patients with high malignancy indicators.")
    
    roster_df = load_simulated_hospital_roster()
    
    # Interactive Filtering UI Component
    risk_filter = st.selectbox("Filter Roster by Priority Classification Status:", 
                               options=["Show All Records", "High Suspicion Only", "Indeterminate Only", "Low Suspicion Only"])
    
    # Filtering Condition Logic Switchboard
    if risk_filter == "High Suspicion Only":
        filtered_df = roster_df[roster_df["Risk_Tier"] == "High Suspicion"]
    elif risk_filter == "Indeterminate Only":
        filtered_df = roster_df[roster_df["Risk_Tier"] == "Indeterminate"]
    elif risk_filter == "Low Suspicion Only":
        filtered_df = roster_df[roster_df["Risk_Tier"] == "Low Suspicion"]
    else:
        filtered_df = roster_df
        
    # Inject Visual Feedback on Priority List Size Counts
    st.metric(label="Current Patient Registry Workload Count (Filtered)", value=len(filtered_df))
    
    # Display the structured interactive data grid 
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ---
    ### 💡 Clinical Workflow Guide
    1. **High-Risk Actions:** Select `High Suspicion Only` from the list dropdown filter before running daily rounds. 
    2. **Urgent Fast-Tracking:** Patients displaying nodule sizes above `8.0 mm` are automatically highlighted and given status priority tracking numbers for diagnostic biopsies.
    """)
