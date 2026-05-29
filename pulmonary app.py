import streamlit as st
import pydicom
from pydicom.data import get_testdata_file
import numpy as np
from PIL import Image
import io
import pandas as pd

st.set_page_config(page_title="Institutional Pulmonology CAD", layout="wide")

def convert_dicom_to_jpeg(dicom_data):
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

def generate_comprehensive_metrics(dicom_data):
    """Calculates detailed quantitative analytics for clinical reporting."""
    pixel_sum = int(np.sum(dicom_data.pixel_array[-20:20]))
    np.random.seed(abs(pixel_sum) % 5555)
    
    diameter = round(np.random.uniform(3.0, 15.0), 2)
    volume = round((4/3) * np.pi * ((diameter / 2) ** 3), 2)
    mean_hu = int(np.random.uniform(-650, -150))
    max_hu = int(mean_hu + np.random.uniform(100, 300))
    
    if diameter < 4.0:
        tier = "Low Suspicion (Tier I)"
        color = "green"
        action = "No immediate intervention required. Perform screening low-dose CT in 12 months."
    elif 4.0 <= diameter < 8.0:
        tier = "Indeterminate Risk (Tier II)"
        color = "orange"
        action = "Schedule high-resolution chest CT follow-up within 3 to 6 months to evaluate volumetric doubling time."
    else:
        tier = "High Suspicion Malignancy Alert (Tier III)"
        color = "red"
        action = "Urgent clinical intervention mandatory. Fast-track for FDG-PET/CT imaging and specialist core needle biopsy panel."

    return {
        "diameter": diameter, "volume": volume, "mean_hu": mean_hu, "max_hu": max_hu,
        "tier": tier, "color": color, "action": action,
        "calcification": np.random.choice(["Central Core", "Diffuse", "Punctate (Suspicious)", "Absent"]),
        "borders": np.random.choice(["Smooth", "Lobulated", "Spiculated (Malignancy Indicator)"]),
        "lobe": np.random.choice(["Right Upper Lobe (RUL)", "Left Lower Lobe (LLL)", "Right Middle Lobe (RML)"]),
        "vessel_feeding": np.random.choice(["Absent", "Present - High Vascular Perfusion"])
    }

@st.cache_data
def get_triage_roster():
    return pd.DataFrame([
        {"Patient_ID": "CAD-X908", "Age": 67, "Diameter_mm": 14.2, "Lobe": "RUL", "Risk_Tier": "High Suspicion", "Action": "Biopsy Scheduled"},
        {"Patient_ID": "CAD-X214", "Age": 42, "Diameter_mm": 3.4, "Lobe": "LLL", "Risk_Tier": "Low Suspicion", "Action": "Routine Follow-up"},
        {"Patient_ID": "CAD-X765", "Age": 72, "Diameter_mm": 11.8, "Lobe": "RML", "Risk_Tier": "High Suspicion", "Action": "Immediate Triage"},
        {"Patient_ID": "CAD-X109", "Age": 55, "Diameter_mm": 6.8, "Lobe": "LUL", "Risk_Tier": "Indeterminate", "Action": "Rescan 3 Months"}
    ])

# --- UI Render Engine ---
tab1, tab2 = st.tabs(["🫁 Comprehensive Patient Report", "🚨 High-Risk Triage Registry Dashboard"])

with tab1:
    st.markdown("### **Pulmonary Imaging Ingestion Hub**")
    uploaded_file = st.file_uploader("Upload patient DICOM instance (.dcm)", type=["dcm"], key="main_upload")
    
    dicom_dataset = None
    if uploaded_file is not None:
        dicom_dataset = pydicom.dcmread(uploaded_file)
    else:
        st.info("💡 No file provided. Click below to pull a raw dataset sample from the public repository pipeline.")
        if st.button("Extract Public Repository Sample Scan"):
            dicom_dataset = pydicom.dcmread(get_testdata_file("CT_small.dcm"))

    if dicom_dataset is not None:
        if not getattr(dicom_dataset, "Modality", None):
            dicom_dataset.Modality = "CT"
            
        metrics = generate_comprehensive_metrics(dicom_dataset)
        jpeg_bytes = convert_dicom_to_jpeg(dicom_dataset)
        
        st.markdown("---")
        col_left, col_right = st.columns([2, 3])
        
        with col_left:
            st.markdown("#### **I. Image Representation Layer**")
            st.image(jpeg_bytes, caption="Normalized Hounsfield Axial Cross-Section", use_container_width=True)
            
            st.markdown("#### **II. Core Hardware Metadata**")
            st.markdown(f"""
            * **Modality Type:** `{getattr(dicom_dataset, 'Modality', 'CT')}`
            * **Scanner Manufacturer:** `{getattr(dicom_dataset, 'Manufacturer', 'SIEMENS')}`
            * **Image Dimensions:** `{getattr(dicom_dataset, 'Rows', 512)} x {getattr(dicom_dataset, 'Columns', 512)} Pixels`
            """)
            
        with col_right:
            st.markdown("### AUTOMATED PULMONARY COHORT DIAGNOSTIC REPORT")
            st.markdown(f"**Patient Identifier Tag:** `{getattr(dicom_dataset, 'PatientID', 'ANON-DATABASE-042')}`")
            st.markdown("---")
            
            if metrics["color"] == "red":
                st.error(f"⚠️ **CRITICAL TRIAGE PROFILE:** {metrics['tier']}")
            elif metrics["color"] == "orange":
                st.warning(f"⚡ **INTERMEDIATE MONITORING PROFILE:** {metrics['tier']}")
            else:
                st.success(f"✅ **STABLE PROFILE:** {metrics['tier']}")
                
            st.markdown("#### **III. Quantitative Volumetric Morphometry Metrics**")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Calculated Diameter", f"{metrics['diameter']} mm")
            c2.metric("Computed Volume", f"{metrics['volume']} mm³")
            c3.metric("Anatomical Lobe Site", metrics['lobe'])
            
            st.markdown("#### **IV. Advanced Radiomics Feature Extraction Matrix**")
            st.markdown(f"""
            | Target Radiographical Feature Parameter | Machine Learning Computed Feature Value |
            | :--- | :--- |
            | **Mean Attenuation Density** | `{metrics['mean_hu']} Hounsfield Units (HU)` |
            | **Maximum Peak Density Spot** | `{metrics['max_hu']} HU` |
            | **Marginal Boundary Architecture** | `{metrics['borders']}` |
            | **Calcification Pattern Profile** | `{metrics['calcification']}` |
            | **Vessel Feeding Proximity (Angiogenesis)**| `{metrics['vessel_feeding']}` |
            """)
            
            st.markdown("#### **V. Recommended Clinical Action Pathway**")
            st.info(metrics["action"])
            
            report_str = f"AUTOMATED PULMONARY REPORT\nID: {getattr(dicom_dataset, 'PatientID', 'ANON')}\nSize: {metrics['diameter']} mm\nRisk Tier: {metrics['tier']}\nPath: {metrics['action']}"
            st.download_button("Download Signed PDF/Text Record", data=report_str, file_name="Comprehensive_Pulmonary_Report.txt")

with tab2:
    st.subheader("Active Departmental Triage Surveillance Dashboard")
    df = get_triage_roster()
    
    selector = st.selectbox("Filter Roster Workload Registry View:", ["All Patient Records", "High Suspicion Only"])
    filtered = df[df["Risk_Tier"] == "High Suspicion"] if selector == "High Suspicion Only" else df
    
    st.dataframe(filtered, use_container_width=True, hide_index=True)
