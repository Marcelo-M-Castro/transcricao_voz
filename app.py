import streamlit as st
from faster_whisper import WhisperModel
from pathlib import Path
import tempfile

# ------------------------
# Configurações do app
# ------------------------
st.set_page_config(page_title="🎙️ Transcrição de WAV", page_icon="🎧", layout="centered")
st.title("🎙️ Transcrição de Áudio (WAV) com Faster-Whisper")

# Seleção de modelo
model_size = st.selectbox(
    "Escolha o modelo",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=2
)
lang_hint = st.text_input("Idioma (ex: pt, en) - deixe vazio para autodetect", value="pt")

# Upload
uploaded_file = st.file_uploader("Envie um arquivo .wav", type=["wav"])

if uploaded_file is not None:
    # Salvar em arquivo temporário
    temp_dir = Path(tempfile.mkdtemp())
    wav_path = temp_dir / uploaded_file.name
    with open(wav_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("🔄 Carregando modelo... isso pode levar um tempo na primeira vez.")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    st.success(f"✅ Modelo {model_size} carregado.")

    st.info("🎧 Transcrevendo o áudio...")
    segments, _ = model.transcribe(str(wav_path), language=None if not lang_hint else lang_hint, vad_filter=True)

    # Monta transcrição + formatos
    lines, srt, vtt = [], [], ["WEBVTT\n"]
    idx = 1

    def fmt_ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        lines.append(txt)
        start, end = fmt_ts(seg.start), fmt_ts(seg.end)
        srt += [str(idx), f"{start} --> {end}", txt, ""]
        vtt += [f"{start.replace(',', '.')} --> {end.replace(',', '.')}", txt, ""]
        idx += 1

    transcript_txt = "\n".join(lines)
    transcript_srt = "\n".join(srt)
    transcript_vtt = "\n".join(vtt)

    # Mostrar resultado
    st.subheader("📝 Transcrição")
    st.text_area("Texto extraído:", transcript_txt, height=300)

    # Botões de download
    st.download_button("📥 Baixar TXT", transcript_txt, file_name="transcricao.txt")
    st.download_button("📥 Baixar SRT", transcript_srt, file_name="transcricao.srt")
    st.download_button("📥 Baixar VTT", transcript_vtt, file_name="transcricao.vtt")
