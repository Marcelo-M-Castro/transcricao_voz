import streamlit as st
import whisper
from pathlib import Path
import tempfile

st.set_page_config(page_title="🎙️ Transcrição de WAV", page_icon="🎧", layout="centered")
st.title("🎙️ Transcrição de Áudio (WAV)")

# Modelo fixo = medium
MODEL_SIZE = "medium"
LANG_HINT = "pt"  # idioma padrão

uploaded_file = st.file_uploader("Envie um arquivo .wav", type=["wav"])

if uploaded_file is not None:
    temp_dir = Path(tempfile.mkdtemp())
    wav_path = temp_dir / uploaded_file.name
    with open(wav_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"🔄 Carregando modelo '{MODEL_SIZE}'...")
    model = whisper.load_model(MODEL_SIZE)

    st.info("🎧 Transcrevendo...")
    result = model.transcribe(str(wav_path), language=None if not LANG_HINT else LANG_HINT)

    transcript_txt = result["text"]

    st.subheader("📝 Transcrição")
    st.text_area("Texto extraído:", transcript_txt, height=300)

    st.download_button("📥 Baixar TXT", transcript_txt, file_name="transcricao.txt")
