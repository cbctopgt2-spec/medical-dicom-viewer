import streamlit as st
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.pyplot as plt

# For 3D rendering
import pyvista as pv
from pyvista import examples

st.set_page_config(page_title="DICOM MPR & Volume Rendering", layout="wide")
st.title("🧠 DICOM Viewer: Axial, Coronal, Sagittal + 3D Rendering")

uploaded_file = st.file_uploader("ডিকম আপলোড করুন (.dcm)", type=["dcm"])

def show_slice(img, title):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

if uploaded_file:
    dcm = pydicom.dcmread(uploaded_file)
    try:
        arr = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)
    except:
        arr = dcm.pixel_array.astype(np.float32)

    # নিশ্চিত হোন তা 3D array
    if arr.ndim == 2:
        st.error("এই DICOM শুধুমাত্র 2D ছবি—3D সিরিজ নেই।")
        st.stop()
    elif arr.ndim != 3:
        st.error(f"অজ্ঞাত আকারের array: shape={arr.shape}")
        st.stop()

    # Normalize
    arr -= arr.min()
    arr /= arr.max()
    arr *= 255
    vol = arr.astype(np.uint8)

    st.success(f"ডেটা লোড হয়েছে: shape={vol.shape}")

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

    # Interactive 3D rendering
    volume = pv.wrap(vol)
    p = pv.Plotter(off_screen=True)
    p.add_volume(volume, cmap="gray", opacity="sigmoid_6")  # window/level adjust করতে পারেন
    p.camera_position = 'iso'
    img = p.show(screenshot=True)

    st.image(img, caption="3D Rendering view (interactive)", use_column_width=True)

else:
    st.info("দয়া করে একটি DICOM সিরিজ আপলোড করুন (৩D) যাতে multiple slices থাকে।")
