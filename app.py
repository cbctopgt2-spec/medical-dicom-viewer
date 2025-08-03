import streamlit as st
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import plotly.graph_objects as go

st.set_page_config(page_title="DICOM MPR & Volume Rendering", layout="wide")
st.title("🧠 DICOM Viewer: Axial, Coronal, Sagittal + 3D Rendering")

# Allow multiple file upload
uploaded_files = st.file_uploader(
    "ডিকম ফাইলগুলো একসাথে আপলোড করুন (.dcm)", 
    type=["dcm"], 
    accept_multiple_files=True
)

def show_slice(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

if uploaded_files:
    # Read all DICOM files
    slices = []
    for f in uploaded_files:
        dcm = pydicom.dcmread(f)
        slices.append(dcm)

    # Sort by InstanceNumber if available
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        st.warning("⚠️ DICOM ফাইলগুলো InstanceNumber দিয়ে sort করা যায়নি। আপলোড করার আগে নিশ্চিত হোন যে সঠিক ক্রমে আছে।")

    # Convert to numpy arrays and stack
    img_arrays = []
    for dcm in slices:
        try:
            arr = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)
        except Exception:
            arr = dcm.pixel_array.astype(np.float32)
        img_arrays.append(arr)

    vol = np.stack(img_arrays, axis=0)

    # Normalize
    vol -= vol.min()
    vol /= vol.max()
    vol *= 255
    vol = vol.astype(np.uint8)

    st.success(f"ডেটা লোড হয়েছে: shape = {vol.shape}")

    # Scrollable slice viewers
    axial_idx = st.slider("Axial Slice (scroll)", 0, vol.shape[0]-1, vol.shape[0]//2, key="ax")
    coronal_idx = st.slider("Coronal Slice", 0, vol.shape[1]-1, vol.shape[1]//2, key="co")
    sagittal_idx = st.slider("Sagittal Slice", 0, vol.shape[2]-1, vol.shape[2]//2, key="sa")

    col1, col2, col3 = st.columns(3)
    with col1:
        show_slice(vol[axial_idx, :, :], f"Axial Slice {axial_idx}")
    with col2:
        show_slice(vol[:, coronal_idx, :], f"Coronal Slice {coronal_idx}")
    with col3:
        show_slice(vol[:, :, sagittal_idx], f"Sagittal Slice {sagittal_idx}")

    st.markdown("---")
    st.markdown("### 🧊 ৩D Volume Rendering")

    # Interactive 3D rendering using Plotly
    x = np.arange(vol.shape[0])
    y = np.arange(vol.shape[1])
    z = np.arange(vol.shape[2])
    
    # Create a 3D surface plot for volume rendering
    fig = go.Figure(data=go.Volume(
        x=np.repeat(x, len(y)*len(z)),
        y=np.tile(np.repeat(y, len(z)), len(x)),
        z=np.tile(z, len(x)*len(y)),
        value=vol.flatten(),
        opacity=0.1,  # Adjust opacity for better visibility
        surface_count=20,
        colorscale="Gray",
        colorbar=dict(title="Intensity")
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title="3D Volume Rendering"
    )

    st.plotly_chart(fig)

else:
    st.info("দয়া করে একাধিক DICOM ফাইল আপলোড করুন যাতে ৩D ভলিউম তৈরি করা যায়।")
