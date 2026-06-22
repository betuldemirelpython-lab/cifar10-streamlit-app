import json
import pathlib
import streamlit as st
import tensorflow as tf
import numpy as np
from utils.preprocess import preprocess_image, get_class_names
from utils.visualize import plot_training_history, plot_prediction_bar

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 CNN Sınıflandırıcı",
    page_icon="🧠",
    layout="wide",
)

# Load class names
class_names = get_class_names()

# Load model with caching
@st.cache_resource
def load_model():
    model_path = pathlib.Path('model') / 'cifar10_cnn.h5'
    if not model_path.is_file():
        st.error("Model bulunamadı. Önce train.py çalıştırın: python train.py")
        st.stop()
    # compile=False avoids loading optimizer state which can cause version mismatches
    return tf.keras.models.load_model(str(model_path), compile=False)

model = load_model()

# Sidebar
st.sidebar.title("🧠 CIFAR-10 CNN")
st.sidebar.info("10 farklı nesne sınıfını tanıyan derin öğrenme modeli")
st.sidebar.info("Sınıflar: " + ", ".join(class_names))

with st.sidebar.expander("📊 Model Bilgisi"):
    st.write(f"Toplam parametre sayısı: {model.count_params():,}")
    st.write(f"Katman sayısı: {len(model.layers)}")

with st.sidebar.expander("📈 Eğitim Geçmişi"):
    history_path = pathlib.Path('model') / 'training_history.json'
    if history_path.is_file():
        with open(history_path, 'r') as f:
            history_dict = json.load(f)
        fig = plot_training_history(history_dict)
        st.pyplot(fig)
    else:
        st.write("Henüz eğitim geçmişi yok")

# Main area
st.title("🧠 CIFAR-10 Görüntü Sınıflandırıcı")
st.caption("CNN modeli ile görüntüleri 10 farklı sınıfa ayırır")
st.divider()

col1, col2 = st.columns(2)

# Column 1 – Image upload / selection
with col1:
    st.subheader("📁 Görüntü Seç")
    tabs = st.tabs(["Yükle", "Örnek Kullan"])
    with tabs[0]:
        uploaded_file = st.file_uploader("Görüntü yükle", type=["jpg", "jpeg", "png", "webp"], key="upload")
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.session_state["current_image"] = image_bytes
            st.image(image_bytes, use_container_width=True)
    with tabs[1]:
        sample_dir = pathlib.Path('assets') / 'sample_images'
        sample_files = []
        if sample_dir.is_dir():
            sample_files = [f for f in sample_dir.iterdir() if f.is_file()]
        sample_options = [f.name for f in sample_files]
        if sample_options:
            choice = st.selectbox("Örnek görüntü seç", ["---"] + sample_options)
            if choice != "---":
                selected_path = sample_dir / choice
                image_bytes = selected_path.read_bytes()
                st.session_state["current_image"] = image_bytes
                st.image(image_bytes, use_container_width=True)
        else:
            st.write("Örnek görüntü mevcut değil.")

# Column 2 – Prediction
with col2:
    st.subheader("🔍 Tahmin Sonucu")
    if st.button("Tahmin Et 🚀", type="primary"):
        if "current_image" not in st.session_state:
            st.warning("Önce bir görüntü seçin")
        else:
            try:
                with st.spinner("Model tahmin yapıyor..."):
                    img_array = preprocess_image(st.session_state["current_image"])
                    predictions = model.predict(img_array)
                    pred_idx = np.argmax(predictions[0])
                    confidence = predictions[0][pred_idx]
                
                st.write("👀 Modelin Gördüğü (32x32):")
                st.image(img_array[0], width=128, clamp=True)
                
                st.metric("🎯 Tahmin", class_names[pred_idx], f"%{confidence*100:.1f}")
                st.success(f"Model bu görüntünün bir {class_names[pred_idx]} olduğunu düşünüyor!")
                st.divider()
                st.caption("Tüm sınıf olasılıkları:")
                fig = plot_prediction_bar(predictions, class_names)
                st.pyplot(fig)
                st.balloons()
            except Exception as e:
                st.error(f"Tahmin sırasında bir hata oluştu: {e}")
