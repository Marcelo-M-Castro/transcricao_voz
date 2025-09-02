import streamlit as st
import whisper
from pathlib import Path
import tempfile

st.set_page_config(page_title="🎙️ Transcrição de WAV", page_icon="🎧", layout="centered")
st.title("🎙️ Transcrição de Áudio (WAV)")

# ------------------------
# Modelo fixo = small (leve para Streamlit Cloud)
# ------------------------
MODEL_SIZE = "small"
LANG_HINT = "pt"

# ------------------------
# Upload de arquivo
# ------------------------
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

    # ------------------------
    # Gerar SRT e VTT
    # ------------------------
    lines, srt, vtt = [], [], ["WEBVTT\n"]
    idx = 1

    def fmt_ts(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    for seg in result.get("segments", []):
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        lines.append(txt)
        start, end = fmt_ts(seg["start"]), fmt_ts(seg["end"])
        srt += [str(idx), f"{start} --> {end}", txt, ""]
        vtt += [f"{start.replace(',', '.')} --> {end.replace(',', '.')}", txt, ""]
        idx += 1

    transcript_srt = "\n".join(srt)
    transcript_vtt = "\n".join(vtt)

    # ------------------------
    # Exibir e permitir download
    # ------------------------
    st.subheader("📝 Transcrição")
    st.text_area("Texto extraído:", transcript_txt, height=300)

    st.download_button("📥 Baixar TXT", transcript_txt, file_name="transcricao.txt")
    st.download_button("📥 Baixar SRT", transcript_srt, file_name="transcricao.srt")
    st.download_button("📥 Baixar VTT", transcript_vtt, file_name="transcricao.vtt")
