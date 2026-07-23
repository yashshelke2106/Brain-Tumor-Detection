"""
Brain Tumor Detection — Streamlit App (Enhanced UI)
Upload a brain MRI image and the CNN model will classify it as
Meningioma, Glioma, or Pituitary tumor with confidence scores
and detailed medical information.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 128
CLASS_NAMES = ['Meningioma', 'Glioma', 'Pituitary']
CLASS_COLORS = ['#e74c3c', '#3498db', '#2ecc71']
CLASS_ICONS = ['🔴', '🔵', '🟢']

CLASS_INFO = {
    'Meningioma': {
        'icon': '🔴',
        'color': '#e74c3c',
        'gradient': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
        'description': 'A tumor that arises from the meninges — the protective membranes surrounding the brain and spinal cord. Most meningiomas are benign (non-cancerous) and slow-growing.',
        'location': 'Surface of the brain, attached to the dura mater',
        'prevalence': '~30% of all primary brain tumors',
        'severity': 'Usually benign (WHO Grade I)',
        'severity_level': 'Low',
        'growth_rate': 'Slow',
        'common_age': '40–70 years',
        'gender_ratio': 'More common in women (2:1)',
        'symptoms': [
            'Headaches that worsen over time',
            'Vision changes or loss',
            'Hearing loss or ringing in ears',
            'Memory loss',
            'Seizures',
            'Weakness in arms or legs'
        ],
        'treatment': [
            'Observation (watch and wait for small tumors)',
            'Surgical removal (craniotomy)',
            'Radiation therapy (stereotactic radiosurgery)',
            'Medications to manage symptoms'
        ],
        'prognosis': '5-year survival rate is approximately 84%. Many meningiomas can be cured with complete surgical removal.',
        'risk_factors': 'Prior radiation exposure, neurofibromatosis type 2, hormonal factors, obesity'
    },
    'Glioma': {
        'icon': '🔵',
        'color': '#3498db',
        'gradient': 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
        'description': 'A tumor originating from glial cells that support nerve cells in the brain. Gliomas are the most common type of primary brain tumor and range from low-grade (slow-growing) to high-grade (aggressive).',
        'location': 'Within the brain tissue (cerebral hemispheres, brainstem, cerebellum)',
        'prevalence': '~33% of all brain tumors, most common primary brain tumor',
        'severity': 'Ranges from Grade I (benign) to Grade IV (GBM — highly malignant)',
        'severity_level': 'Variable (Low to High)',
        'growth_rate': 'Variable — slow (low-grade) to rapid (high-grade)',
        'common_age': '45–65 years (high-grade); 20–40 years (low-grade)',
        'gender_ratio': 'Slightly more common in men',
        'symptoms': [
            'Persistent headaches (often worse in the morning)',
            'Seizures (often the first symptom)',
            'Cognitive or personality changes',
            'Nausea and vomiting',
            'Speech difficulties',
            'Balance and coordination problems',
            'Vision problems'
        ],
        'treatment': [
            'Maximal safe surgical resection',
            'Radiation therapy (external beam, proton therapy)',
            'Chemotherapy (Temozolomide is standard)',
            'Targeted therapy and immunotherapy (emerging)',
            'Tumor Treating Fields (TTFields) for GBM'
        ],
        'prognosis': 'Varies widely by grade. Low-grade: 5-year survival ~80%. High-grade (GBM): median survival 12-18 months with treatment.',
        'risk_factors': 'Ionizing radiation exposure, genetic syndromes (Li-Fraumeni, NF1), family history'
    },
    'Pituitary': {
        'icon': '🟢',
        'color': '#2ecc71',
        'gradient': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
        'description': 'A tumor developing in the pituitary gland — a pea-sized organ at the base of the brain that controls hormone production. Most pituitary tumors are benign adenomas.',
        'location': 'Pituitary gland (sella turcica, base of brain behind the nose)',
        'prevalence': '~15% of all intracranial neoplasms',
        'severity': 'Usually benign (adenoma)',
        'severity_level': 'Low',
        'growth_rate': 'Slow',
        'common_age': '30–60 years',
        'gender_ratio': 'Equal distribution',
        'symptoms': [
            'Hormonal imbalances (excess or deficiency)',
            'Vision problems (bitemporal hemianopia)',
            'Headaches',
            'Unexplained weight gain or loss',
            'Fatigue and weakness',
            'Menstrual irregularities / erectile dysfunction',
            'Excessive thirst and urination'
        ],
        'treatment': [
            'Transsphenoidal surgery (through the nose)',
            'Medication (dopamine agonists for prolactinomas)',
            'Radiation therapy (for residual or recurrent tumors)',
            'Hormone replacement therapy',
            'Observation for incidental findings'
        ],
        'prognosis': 'Excellent prognosis. Most can be managed effectively with medication or surgery. 5-year survival rate exceeds 95%.',
        'risk_factors': 'Multiple endocrine neoplasia type 1 (MEN1), Carney complex, family history'
    }
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.3rem;
        position: relative;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.7);
        position: relative;
        font-weight: 300;
    }

    .result-banner {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    .result-banner::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 150px; height: 150px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }
    .result-label { font-size: 0.9rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
    .result-name { font-size: 2.5rem; font-weight: 800; margin: 0.3rem 0; letter-spacing: -0.02em; }
    .result-conf { font-size: 1.4rem; font-weight: 300; opacity: 0.9; }

    .confidence-meter {
        background: #f0f2f5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e1e5eb;
        transition: all 0.3s ease;
    }
    .confidence-meter:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .conf-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .conf-label { font-weight: 600; font-size: 1rem; color: #333; }
    .conf-value { font-weight: 700; font-size: 1.1rem; }
    .conf-bar-bg {
        width: 100%;
        height: 10px;
        background: #dfe3e8;
        border-radius: 5px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-out;
    }

    .info-section {
        background: white;
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        box-shadow: 0 2px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef0f2;
    }
    .info-section h3 {
        margin-top: 0;
        color: #1a1a2e;
        font-weight: 700;
    }

    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 1px solid #eef0f2;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .stat-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
    .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .stat-value { font-size: 1.1rem; font-weight: 700; color: #333; margin-top: 0.2rem; }

    .symptom-tag {
        display: inline-block;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #555;
    }

    .treatment-item {
        background: #f8fffe;
        border-left: 3px solid #2ecc71;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    .disclaimer-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        font-size: 0.85rem;
        color: #856404;
        margin: 1rem 0;
    }

    .upload-zone {
        border: 2px dashed #cbd5e0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #fafbfc;
        transition: all 0.3s;
    }
    .upload-zone:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }

    .footer-text {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_paths = [
        'brain_tumor_cnn_model.h5',
        'brain_tumor_cnn_savedmodel',
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path)
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
    return None


def preprocess_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    img_resized = resize(img_array, (IMG_SIZE, IMG_SIZE, 3),
                         anti_aliasing=True, preserve_range=True).astype(np.float32)
    return img_resized / 255.0


# ─────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────
def create_gauge_chart(confidence, color, name):
    """Semi-circular gauge showing confidence."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([])
    ax.set_thetagrids([])
    ax.spines['polar'].set_visible(False)

    # Background arc
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.6, 1.0, color='#ecf0f1', alpha=0.5)

    # Confidence arc
    theta_val = np.linspace(0, np.pi * (confidence / 100), 100)
    ax.fill_between(theta_val, 0.6, 1.0, color=color, alpha=0.8)

    # Center text
    ax.text(np.pi / 2, 0.15, f'{confidence:.1f}%', ha='center', va='center',
            fontsize=22, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(np.pi / 2, 0.02, name, ha='center', va='center',
            fontsize=10, color='#666',
            transform=ax.transAxes)

    plt.tight_layout()
    return fig


def create_confidence_bars(predictions):
    """Horizontal gradient bars with percentages."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    confidences = predictions[0] * 100
    y_pos = np.arange(len(CLASS_NAMES))

    for i, (conf, color, name) in enumerate(zip(confidences, CLASS_COLORS, CLASS_NAMES)):
        # Shadow bar
        ax.barh(i, 100, color='#f0f2f5', height=0.55, zorder=1)
        # Value bar with rounded ends
        bar = ax.barh(i, conf, color=color, height=0.55, zorder=2,
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        # Percentage text
        if conf > 15:
            ax.text(conf - 2, i, f'{conf:.1f}%', va='center', ha='right',
                    fontsize=12, fontweight='bold', color='white', zorder=3)
        else:
            ax.text(conf + 2, i, f'{conf:.1f}%', va='center', ha='left',
                    fontsize=12, fontweight='bold', color=color, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{icon}  {name}' for icon, name in zip(CLASS_ICONS, CLASS_NAMES)],
                       fontsize=13, fontweight='600')
    ax.set_xlim([0, 105])
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.invert_yaxis()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def create_radar_chart(predictions):
    """Radar / spider chart of class probabilities."""
    labels = CLASS_NAMES
    values = list(predictions[0] * 100)
    values += values[:1]  # close the polygon

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    ax.fill(angles, values, color='#667eea', alpha=0.15)
    ax.plot(angles, values, color='#667eea', linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='white', markeredgecolor='#667eea',
            markeredgewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{icon} {name}' for icon, name in zip(CLASS_ICONS, labels)],
                       fontsize=11, fontweight='600')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='#999')
    ax.spines['polar'].set_color('#ddd')
    ax.grid(color='#eee', linewidth=0.8)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def create_donut_chart(predictions):
    """Donut chart with center label."""
    fig, ax = plt.subplots(figsize=(5, 5))
    confidences = predictions[0] * 100
    pred_idx = np.argmax(confidences)

    wedges, texts, autotexts = ax.pie(
        confidences, labels=None, colors=CLASS_COLORS,
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=3),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    for i, autotext in enumerate(autotexts):
        autotext.set_color('white' if confidences[i] > 15 else CLASS_COLORS[i])

    # Center circle
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)

    # Center text
    ax.text(0, 0.08, CLASS_NAMES[pred_idx], ha='center', va='center',
            fontsize=14, fontweight='bold', color=CLASS_COLORS[pred_idx])
    ax.text(0, -0.12, f'{confidences[pred_idx]:.1f}%', ha='center', va='center',
            fontsize=20, fontweight='800', color=CLASS_COLORS[pred_idx])

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=f'{icon} {n}')
                      for c, icon, n in zip(CLASS_COLORS, CLASS_ICONS, CLASS_NAMES)]
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=9, frameon=False)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Brain Tumor Detection</div>
    <div class="hero-subtitle">CNN-powered MRI classification  ·  Meningioma  ·  Glioma  ·  Pituitary</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 How It Works")
    st.markdown("""
    1. **Upload** a brain MRI image
    2. The CNN **preprocesses** it to 128×128
    3. The model **classifies** the tumor type
    4. View **confidence scores** and **tumor details**
    """)

    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    st.code("""
Conv2D(32) → BN → Pool
Conv2D(64) → BN → Pool
Conv2D(128) → BN → Pool
Flatten → Dense(256)
Dropout → Dense(3, softmax)
    """, language=None)

    st.markdown("---")
    st.markdown("### 📊 Training Stats")
    col1, col2 = st.columns(2)
    col1.metric("Images", "3,064")
    col2.metric("Classes", "3")
    col1.metric("Input Size", "128×128")
    col2.metric("Framework", "TF/Keras")

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>Medical Disclaimer</strong><br>
    This tool is for <strong>educational and research purposes only</strong>.
    It is NOT a substitute for professional medical diagnosis.
    Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
model = load_model()

if model is None:
    st.error("""
    **⚠️ Model file not found!**

    Train the model first:
    1. Open `Brain_Tumor_CNN_Project.ipynb`
    2. Run all cells → saves `brain_tumor_cnn_model.h5`
    3. Relaunch: `streamlit run app.py`
    """)
    st.stop()

# ─────────────────────────────────────────────
# Upload section — multi-image + folder support
# ─────────────────────────────────────────────
st.markdown("### 📤 Upload MRI Scans")

upload_tab1, upload_tab2, upload_tab3 = st.tabs([
    "📎 Upload Images", "📂 Load from Folder", "🧪 Sample Test Images"
])

uploaded_files_main = []

with upload_tab1:
    st.markdown("Upload **one or multiple** MRI images at once.")
    raw_uploads = st.file_uploader(
        "Choose brain MRI image(s)",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Select one or many MRI scans — hold Ctrl/Cmd to pick multiple files",
        key="main_uploader"
    )
    if raw_uploads:
        uploaded_files_main = [('upload', f.name, f) for f in raw_uploads]

with upload_tab2:
    st.markdown("Enter a **folder path** on your computer containing MRI images.")
    folder_path = st.text_input(
        "Folder path",
        placeholder=r"e.g.  C:\Users\yashs\Documents\mri_scans",
        key="folder_input"
    )
    if folder_path and os.path.isdir(folder_path):
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        folder_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_ext)
        ])
        if folder_files:
            st.success(f"Found **{len(folder_files)}** images in folder")
            uploaded_files_main = [
                ('folder', f, os.path.join(folder_path, f)) for f in folder_files
            ]
        else:
            st.warning("No image files (jpg/png/bmp/tiff) found in this folder.")
    elif folder_path:
        st.error("Folder path does not exist. Check the path and try again.")

with upload_tab3:
    test_dir = 'test_images-20210704T210303Z-001/test_images'
    if os.path.exists(test_dir):
        valid_ext = ('.jpg', '.jpeg', '.png')
        test_files = sorted([
            f for f in os.listdir(test_dir) if f.lower().endswith(valid_ext)
        ])
        if test_files:
            sample_mode = st.radio(
                "Select mode:",
                ["Pick individual images", "Load all test images"],
                horizontal=True,
                key="sample_mode"
            )
            if sample_mode == "Load all test images":
                uploaded_files_main = [
                    ('folder', f, os.path.join(test_dir, f)) for f in test_files
                ]
                st.info(f"Loaded all **{len(test_files)}** sample images")
            else:
                selected_samples = st.multiselect(
                    "Pick one or more sample images:",
                    test_files,
                    key="sample_picker"
                )
                if selected_samples:
                    uploaded_files_main = [
                        ('folder', f, os.path.join(test_dir, f)) for f in selected_samples
                    ]
        else:
            st.info("No sample images found in test directory.")
    else:
        st.info("Test image directory not found.")


# ─────────────────────────────────────────────
# Helper: open image from either source
# ─────────────────────────────────────────────
def open_image(source_type, name, source):
    """Open an image from upload object or file path."""
    if source_type == 'upload':
        return Image.open(source)
    else:
        return Image.open(source)


# ─────────────────────────────────────────────
# Prediction + Results
# ─────────────────────────────────────────────
if len(uploaded_files_main) == 1:
    # ── Single image mode: full detailed view ──
    src_type, src_name, src_obj = uploaded_files_main[0]
    image = open_image(src_type, src_name, src_obj)
    image_source = src_name
    st.markdown("---")

    with st.spinner("🔬 Analyzing MRI scan..."):
        img_processed = preprocess_image(image)
        predictions = model.predict(img_processed[np.newaxis, ...], verbose=0)

    predicted_class = np.argmax(predictions[0])
    predicted_name = CLASS_NAMES[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    info = CLASS_INFO[predicted_name]

    # ── Result banner ──
    st.markdown(f"""
    <div class="result-banner" style="background: {info['gradient']};">
        <div class="result-label">Detection Result</div>
        <div class="result-name">{info['icon']}  {predicted_name} Tumor</div>
        <div class="result-conf">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Image + Confidence meters ──
    col_img, col_conf = st.columns([1, 1.3])

    with col_img:
        st.markdown("#### 🖼️ MRI Scan")
        st.image(image, caption=f"Source: {image_source}", use_container_width=True)

        with st.expander("🔍 Preprocessed View (model input)"):
            fig_pp, ax_pp = plt.subplots(1, 2, figsize=(8, 3.5))
            ax_pp[0].imshow(np.array(image))
            ax_pp[0].set_title('Original', fontweight='bold', fontsize=11)
            ax_pp[0].axis('off')
            ax_pp[1].imshow(img_processed)
            ax_pp[1].set_title(f'Resized {IMG_SIZE}×{IMG_SIZE} + Normalized', fontweight='bold', fontsize=11)
            ax_pp[1].axis('off')
            fig_pp.patch.set_facecolor('white')
            plt.tight_layout()
            st.pyplot(fig_pp)
            plt.close()

    with col_conf:
        st.markdown("#### 📊 Classification Scores")

        # Styled confidence meters
        for i, (name, color, icon) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, CLASS_ICONS)):
            prob = predictions[0][i] * 100
            is_predicted = (i == predicted_class)
            border = f"border: 2px solid {color};" if is_predicted else ""
            bg = f"background: linear-gradient(90deg, {color}11 0%, #f0f2f5 100%);" if is_predicted else ""
            badge = f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-size:0.7rem;font-weight:700;margin-left:8px;">DETECTED</span>' if is_predicted else ''

            st.markdown(f"""
            <div class="confidence-meter" style="{border}{bg}">
                <div class="conf-header">
                    <span class="conf-label">{icon} {name}{badge}</span>
                    <span class="conf-value" style="color:{color};">{prob:.1f}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{prob}%;background:{color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts row ──
    st.markdown("---")
    st.markdown("### 📈 Visual Analysis")
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Donut Chart"])

    with tab1:
        fig_bar = create_confidence_bars(predictions)
        st.pyplot(fig_bar)
        plt.close()
    with tab2:
        fig_radar = create_radar_chart(predictions)
        st.pyplot(fig_radar)
        plt.close()
    with tab3:
        fig_donut = create_donut_chart(predictions)
        st.pyplot(fig_donut)
        plt.close()

    # ── Tumor Information Section ──
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; margin:1rem 0;">
        <span style="font-size:2.5rem;">{info['icon']}</span>
        <h2 style="margin:0.3rem 0 0.2rem; color:{info['color']};">About {predicted_name} Tumors</h2>
        <p style="color:#888; font-size:0.95rem;">{info['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row — all in one HTML block
    stat_items = [
        ("📍", "Location", info['location']),
        ("📊", "Prevalence", info['prevalence']),
        ("⚡", "Growth Rate", info['growth_rate']),
        ("🎂", "Common Age", info['common_age']),
        ("👥", "Gender Ratio", info['gender_ratio']),
        ("🛡️", "Severity", info['severity_level']),
    ]

    stat_cards_html = ''.join(
        f'<div style="flex:1;min-width:120px;background:white;border-radius:14px;'
        f'padding:1rem 0.6rem;text-align:center;box-shadow:0 2px 15px rgba(0,0,0,0.05);'
        f'border:1px solid #eef0f2;">'
        f'<div style="font-size:1.8rem;margin-bottom:0.3rem;">{icon}</div>'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;">{label}</div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:#333;margin-top:0.2rem;">'
        f'{value}</div></div>'
        for icon, label, value in stat_items
    )

    st.markdown(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin:1rem 0 1.5rem;">
        {stat_cards_html}
    </div>
    """, unsafe_allow_html=True)

    # Symptoms + Treatment
    col_sym, col_treat = st.columns([1, 1])

    # Build symptoms HTML as one block
    symptoms_html = ''.join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:10px 14px;'
        f'margin:6px 0;background:#fafafa;border-radius:10px;border:1px solid #eee;">'
        f'<span style="color:{info["color"]};font-size:1.1rem;flex-shrink:0;">●</span>'
        f'<span style="color:#333;font-size:0.9rem;line-height:1.4;">{s}</span></div>'
        for s in info['symptoms']
    )
    with col_sym:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                🩺 Common Symptoms</h3>
            {symptoms_html}
        </div>
        """, unsafe_allow_html=True)

    # Build treatment HTML as one block
    treatments_html = ''.join(
        f'<div style="background:#f0faf6;border-left:4px solid {info["color"]};'
        f'padding:12px 16px;margin:8px 0;border-radius:0 10px 10px 0;'
        f'font-size:0.9rem;color:#333;line-height:1.5;">'
        f'<span style="font-weight:600;color:{info["color"]};margin-right:6px;">▸</span>'
        f'{t}</div>'
        for t in info['treatment']
    )
    with col_treat:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                💊 Treatment Options</h3>
            {treatments_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prognosis + Risk factors (also single blocks)
    col_prog, col_risk = st.columns([1, 1])
    with col_prog:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                📈 Prognosis</h3>
            <p style="font-size:0.95rem;color:#333;line-height:1.7;margin:0;">
                {info['prognosis']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_risk:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                ⚠️ Risk Factors</h3>
            <p style="font-size:0.95rem;color:#333;line-height:1.7;margin:0;">
                {info['risk_factors']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Medical disclaimer
    st.markdown("""
    <div class="disclaimer-box" style="margin-top:1.5rem;">
        ⚠️ <strong>Important:</strong> The information above is for educational reference only.
        Tumor characteristics vary significantly between individual cases.
        Diagnosis, treatment, and prognosis should be determined by qualified medical professionals
        using comprehensive clinical evaluation, imaging, and histopathological analysis.
    </div>
    """, unsafe_allow_html=True)

elif len(uploaded_files_main) > 1:
    # ── Multi-image batch mode ──
    import pandas as pd

    st.markdown("---")
    n_images = len(uploaded_files_main)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                padding:1.5rem 2rem;border-radius:14px;color:white;text-align:center;
                margin-bottom:1.5rem;">
        <div style="font-size:1.8rem;font-weight:700;">📁 Batch Analysis</div>
        <div style="opacity:0.85;">Processing <strong>{n_images}</strong> MRI images</div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Starting analysis...")

    results = []
    all_images = []

    for idx, (src_type, src_name, src_obj) in enumerate(uploaded_files_main):
        img = open_image(src_type, src_name, src_obj)
        img_proc = preprocess_image(img)
        pred = model.predict(img_proc[np.newaxis, ...], verbose=0)
        pred_idx = np.argmax(pred)
        pred_class = CLASS_NAMES[pred_idx]
        pred_icon = CLASS_ICONS[pred_idx]
        pred_color = CLASS_COLORS[pred_idx]
        conf = pred[0].max() * 100

        results.append({
            'File': src_name,
            'Prediction': pred_class,
            'Icon': pred_icon,
            'Color': pred_color,
            'Confidence': conf,
            'Probabilities': pred[0]
        })
        all_images.append(img)

        progress_bar.progress(
            (idx + 1) / n_images,
            text=f"Analyzed {idx + 1}/{n_images}: {src_name}"
        )

    progress_bar.empty()
    st.success(f"Completed analysis of **{n_images}** images!")

    # ── Summary stats ──
    counts = {n: 0 for n in CLASS_NAMES}
    for r in results:
        counts[r['Prediction']] += 1
    avg_conf = np.mean([r['Confidence'] for r in results])

    summary_html = (
        f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin:1rem 0;">'
        f'<div style="flex:1;min-width:140px;background:white;border-radius:14px;'
        f'padding:1.2rem;text-align:center;box-shadow:0 2px 15px rgba(0,0,0,0.05);'
        f'border:1px solid #eef0f2;">'
        f'<div style="font-size:2rem;">🧠</div>'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;">Total Scans</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#333;">{n_images}</div></div>'
        f'<div style="flex:1;min-width:140px;background:white;border-radius:14px;'
        f'padding:1.2rem;text-align:center;box-shadow:0 2px 15px rgba(0,0,0,0.05);'
        f'border:1px solid #eef0f2;">'
        f'<div style="font-size:2rem;">🎯</div>'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;">Avg Confidence</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#333;">{avg_conf:.1f}%</div></div>'
    )
    for name, icon, color in zip(CLASS_NAMES, CLASS_ICONS, CLASS_COLORS):
        summary_html += (
            f'<div style="flex:1;min-width:140px;background:white;border-radius:14px;'
            f'padding:1.2rem;text-align:center;box-shadow:0 2px 15px rgba(0,0,0,0.05);'
            f'border:1px solid #eef0f2;border-top:3px solid {color};">'
            f'<div style="font-size:2rem;">{icon}</div>'
            f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
            f'letter-spacing:1px;font-weight:600;">{name}</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{color};">'
            f'{counts[name]}</div></div>'
        )
    summary_html += '</div>'
    st.markdown(summary_html, unsafe_allow_html=True)

    # ── Distribution pie chart ──
    st.markdown("---")
    col_pie, col_table = st.columns([1, 1.5])

    with col_pie:
        st.markdown("#### 📊 Class Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(5, 5))
        dist_vals = [counts[n] for n in CLASS_NAMES]
        if sum(dist_vals) > 0:
            wedges, texts, autotexts = ax_dist.pie(
                dist_vals, labels=CLASS_NAMES, colors=CLASS_COLORS,
                autopct=lambda p: f'{p:.0f}%\n({int(round(p * n_images / 100))})',
                startangle=90, textprops={'fontsize': 10},
                wedgeprops=dict(width=0.45, edgecolor='white', linewidth=3),
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontweight('bold')
                at.set_fontsize(9)
            centre = plt.Circle((0, 0), 0.5, fc='white')
            ax_dist.add_artist(centre)
            ax_dist.text(0, 0.05, 'Batch', ha='center', va='center',
                         fontsize=13, fontweight='bold', color='#333')
            ax_dist.text(0, -0.12, f'{n_images} scans', ha='center', va='center',
                         fontsize=10, color='#888')
        ax_dist.set_facecolor('white')
        fig_dist.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close()

    with col_table:
        st.markdown("#### 📋 Results Table")
        df = pd.DataFrame([{
            'File': r['File'],
            'Prediction': f"{r['Icon']} {r['Prediction']}",
            'Confidence': f"{r['Confidence']:.1f}%",
            'Meningioma': f"{r['Probabilities'][0]*100:.1f}%",
            'Glioma': f"{r['Probabilities'][1]*100:.1f}%",
            'Pituitary': f"{r['Probabilities'][2]*100:.1f}%",
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    # ── Image grid with predictions ──
    st.markdown("---")
    st.markdown("#### 🖼️ All Predictions")

    n_cols_grid = min(5, n_images)
    grid_cols = st.columns(n_cols_grid)

    for i, (r, img) in enumerate(zip(results, all_images)):
        with grid_cols[i % n_cols_grid]:
            st.image(img,
                     caption=f"{r['Icon']} {r['Prediction']} ({r['Confidence']:.0f}%)",
                     use_container_width=True)

    # ── Click any image to see details ──
    st.markdown("---")
    st.markdown("#### 🔍 Detailed View — Select an Image")

    detail_options = [f"{r['Icon']} {r['File']} → {r['Prediction']} ({r['Confidence']:.1f}%)"
                      for r in results]
    selected_detail = st.selectbox("Pick an image for full details:", detail_options,
                                   key="detail_selector")
    detail_idx = detail_options.index(selected_detail)
    detail_r = results[detail_idx]
    detail_img = all_images[detail_idx]
    detail_info = CLASS_INFO[detail_r['Prediction']]

    col_d1, col_d2 = st.columns([1, 1.3])
    with col_d1:
        st.image(detail_img, caption=detail_r['File'], use_container_width=True)

    with col_d2:
        st.markdown(f"""
        <div style="background:{detail_info['gradient']};padding:1.5rem;border-radius:14px;
                    color:white;margin-bottom:1rem;">
            <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:2px;
                        opacity:0.8;font-weight:600;">Detected</div>
            <div style="font-size:1.8rem;font-weight:800;">{detail_info['icon']} {detail_r['Prediction']}</div>
            <div style="font-size:1.2rem;font-weight:300;opacity:0.9;">
                Confidence: {detail_r['Confidence']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars for this image
        for ci, (cn, cc, cicon) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, CLASS_ICONS)):
            cp = detail_r['Probabilities'][ci] * 100
            is_det = (cn == detail_r['Prediction'])
            bdr = f"border:2px solid {cc};" if is_det else ""
            bg = f"background:linear-gradient(90deg,{cc}11 0%,#f0f2f5 100%);" if is_det else ""
            badge = (f'<span style="background:{cc};color:white;padding:2px 10px;'
                     f'border-radius:10px;font-size:0.7rem;font-weight:700;'
                     f'margin-left:8px;">DETECTED</span>') if is_det else ''
            st.markdown(f"""
            <div style="background:#f0f2f5;border-radius:12px;padding:0.9rem 1.2rem;
                        margin:0.4rem 0;border:1px solid #e1e5eb;{bdr}{bg}">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:0.5rem;">
                    <span style="font-weight:600;font-size:0.95rem;color:#333;">
                        {cicon} {cn}{badge}</span>
                    <span style="font-weight:700;font-size:1rem;color:{cc};">{cp:.1f}%</span>
                </div>
                <div style="width:100%;height:8px;background:#dfe3e8;border-radius:4px;overflow:hidden;">
                    <div style="width:{cp}%;height:100%;background:{cc};border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tumor info for selected image
    st.markdown("<br>", unsafe_allow_html=True)

    det_symptoms_html = ''.join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:10px 14px;'
        f'margin:6px 0;background:#fafafa;border-radius:10px;border:1px solid #eee;">'
        f'<span style="color:{detail_info["color"]};font-size:1.1rem;flex-shrink:0;">●</span>'
        f'<span style="color:#333;font-size:0.9rem;line-height:1.4;">{s}</span></div>'
        for s in detail_info['symptoms']
    )
    det_treatments_html = ''.join(
        f'<div style="background:#f0faf6;border-left:4px solid {detail_info["color"]};'
        f'padding:12px 16px;margin:8px 0;border-radius:0 10px 10px 0;'
        f'font-size:0.9rem;color:#333;line-height:1.5;">'
        f'<span style="font-weight:600;color:{detail_info["color"]};margin-right:6px;">▸</span>'
        f'{t}</div>'
        for t in detail_info['treatment']
    )

    col_ds, col_dt = st.columns([1, 1])
    with col_ds:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                🩺 Common Symptoms</h3>
            {det_symptoms_html}
        </div>
        """, unsafe_allow_html=True)
    with col_dt:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                💊 Treatment Options</h3>
            {det_treatments_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_dp, col_dr = st.columns([1, 1])
    with col_dp:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                📈 Prognosis</h3>
            <p style="font-size:0.95rem;color:#333;line-height:1.7;margin:0;">
                {detail_info['prognosis']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_dr:
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    box-shadow:0 2px 20px rgba(0,0,0,0.06);border:1px solid #eef0f2;">
            <h3 style="margin-top:0;color:#1a1a2e;font-weight:700;font-size:1.15rem;">
                ⚠️ Risk Factors</h3>
            <p style="font-size:0.95rem;color:#333;line-height:1.7;margin:0;">
                {detail_info['risk_factors']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box" style="margin-top:1.5rem;">
        ⚠️ <strong>Important:</strong> This information is for educational reference only.
        Always consult qualified medical professionals for diagnosis and treatment decisions.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#aaa;">
        <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
        <h3 style="color:#888;">Upload MRI scans to get started</h3>
        <p>Upload single or multiple images, load a folder, or use sample test images</p>
        <p style="font-size:0.85rem;">Supported formats: JPG, PNG, BMP, TIFF</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Tumor Types We Detect")

    overview_cards = ''.join(
        f'<div style="flex:1;min-width:250px;background:white;border-radius:14px;'
        f'padding:1.5rem;text-align:center;box-shadow:0 2px 15px rgba(0,0,0,0.05);'
        f'border:1px solid #eef0f2;border-top:4px solid {CLASS_INFO[n]["color"]};">'
        f'<div style="font-size:2.5rem;margin-bottom:0.5rem;">{CLASS_INFO[n]["icon"]}</div>'
        f'<h3 style="color:{CLASS_INFO[n]["color"]};margin:0.3rem 0;">{n}</h3>'
        f'<p style="font-size:0.85rem;color:#666;line-height:1.5;">'
        f'{CLASS_INFO[n]["description"][:130]}...</p>'
        f'<div style="margin-top:0.8rem;">'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;">Severity</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#333;margin-top:0.2rem;">'
        f'{CLASS_INFO[n]["severity_level"]}</div></div>'
        f'<div style="margin-top:0.6rem;">'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:1px;font-weight:600;">Growth</div>'
        f'<div style="font-size:1rem;font-weight:700;color:#333;margin-top:0.2rem;">'
        f'{CLASS_INFO[n]["growth_rate"]}</div></div></div>'
        for n in CLASS_NAMES
    )

    st.markdown(f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin:1rem 0;">
        {overview_cards}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Brain Tumor Detection CNN  ·  Capstone Project  ·  Innomatics Research Labs<br>
    Built by Yash Shelke  ·  2026
</div>
""", unsafe_allow_html=True)
