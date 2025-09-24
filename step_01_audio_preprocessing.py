
"""
아키텍처의 processing 단계를 구현하는 방법을 단계별로 제시하겠습니다. 이 단계는 AI 기반 회의록 요약 시스템에서 음성 입력부터 최종 문서 출력까지의 주요 처리 과정에 해당합니다.
(중략)
"""

# 1. 필요 라이브러리 설치 및 임포트
import os
import subprocess
import sys

def install_and_import():
    libraries = {
        "librosa": "librosa",
        "soundfile": "sf",
        "numpy": "np",
        "torch": "torch",
        "pyannote.audio": "pyannote.audio",
        "dotenv": "dotenv",
        "transformers": "transformers",
        "accelerate": "accelerate"
    }
    try:
        for lib in libraries:
            __import__(lib.split('.')[0])
    except ImportError:
        print(f"필요한 라이브러리를 설치합니다: {list(libraries.keys())}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + list(libraries.keys())
        )
    
    # 전역 네임스페이스에 라이브러리 임포트
    globals()["librosa"] = __import__("librosa")
    globals()["sf"] = __import__("soundfile")
    globals()["np"] = __import__("numpy")
    globals()["torch"] = __import__("torch")
    from pyannote.audio import Pipeline
    globals()["Pipeline"] = Pipeline
    from dotenv import load_dotenv
    globals()["load_dotenv"] = load_dotenv
    from transformers import pipeline as hf_pipeline
    globals()["hf_pipeline"] = hf_pipeline


install_and_import()

# 2. 전역 설정 및 환경 변수 로드
os.load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"실행 디바이스: {DEVICE}")

# 3. 기능별 함수 정의

# --- 1단계: 전처리 ---
def preprocess_audio(input_path, output_path):
    print(f"\n--- 1단계: 음성 전처리 시작 ---")
    y, sr = librosa.load(input_path, sr=None)
    
    # 노이즈 제거
    noise_profile = np.mean(np.abs(librosa.stft(y[:sr])), axis=1)
    stft_signal = librosa.stft(y)
    stft_denoised = stft_signal - 1.0 * noise_profile[:, np.newaxis]
    stft_denoised = np.maximum(0, np.abs(stft_denoised)) * np.exp(1j * np.angle(stft_signal))
    y_denoised = librosa.istft(stft_denoised)

    # 음량 정규화
    rms = np.sqrt(np.mean(y_denoised**2))
    if rms > 0:
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - 20 * np.log10(rms)
        y_normalized = y_denoised * (10**(change_in_dBFS / 20))
    else:
        y_normalized = y_denoised

    sf.write(output_path, y_normalized, sr)
    print(f"전처리 완료. 결과 파일: {output_path}")
    return output_path, sr

# --- 2단계: 화자 분리 ---
def diarize_speakers(audio_path):
    print(f"\n--- 2단계: 화자 분리 시작 ---")
    if not HUGGINGFACE_TOKEN:
        raise ValueError(".env 파일에 HUGGINGFACEHUB_API_TOKEN을 설정해야 합니다.")
        
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    ).to(DEVICE)
    
    diarization = diarization_pipeline(audio_path)
    print("화자 분리 완료.")
    return diarization

# --- 3단계: 음성 인식 (STT) ---
def transcribe_segments(audio_path, diarization_result):
    print(f"\n--- 3단계: 음성 인식 시작 ---")
    stt_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=DEVICE
    )
    
    transcript = []
    y, sr = librosa.load(audio_path, sr=16000) # Whisper는 16kHz로 처리

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        segment = y[int(start * sr):int(end * sr)]
        
        text = stt_pipeline(
            segment,
            generate_kwargs={"language": "korean"}
        )["text"]
        
        transcript.append({
            "speaker": speaker,
            "start": f"{start:.2f}s",
            "end": f"{end:.2f}s",
            "text": text.strip()
        })
        print(f"[{speaker}] {start:.2f}s - {end:.2f}s: {text.strip()}")
        
    print("음성 인식 완료.")
    return transcript

# --- 4단계: 요약 ---
def summarize_text(full_text):
    print(f"\n--- 4단계: 텍스트 요약 시작 ---")
    summarizer = hf_pipeline(
        "summarization",
        model="gogamza/kobart-summarization",
        device=DEVICE
    )
    
    # 모델의 최대 길이에 맞춰 텍스트 분할
    max_chunk_length = 512
    chunks = [full_text[i:i+max_chunk_length] for i in range(0, len(full_text), max_chunk_length)]
    
    summary = summarizer(chunks, max_length=150, min_length=30, do_sample=False)
    full_summary = " ".join([s['summary_text'] for s in summary])
    
    print("텍스트 요약 완료.")
    return full_summary

# 4. 메인 실행 함수
def main(input_audio_path):
    if not os.path.exists(input_audio_path):
        print(f"오류: 입력 파일 '{input_audio_path}'을(를) 찾을 수 없습니다.")
        return

    # --- 1단계 ---
    processed_audio_path, sr = preprocess_audio(
        input_audio_path,
        "data/processed_audio.wav"
    )

    # --- 2단계 ---
    diarization_result = diarize_speakers(processed_audio_path)

    # --- 3단계 ---
    transcript = transcribe_segments(processed_audio_path, diarization_result)

    # --- 4/5단계 ---
    full_transcript_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
    summary = summarize_text(full_transcript_text)

    print("\n\n================ 회의록 최종 결과 ================")
    print("### 대화 내용 ###")
    for t in transcript:
        print(f"[{t['start']} - {t['end']}] {t['speaker']}: {t['text']}")
    
    print("\n### 회의 요약 ###")
    print(summary)
    print("================================================")


if __name__ == '__main__':
    # --- 실행 설정 ---
    # 여기에 처리할 음성 파일 경로를 입력하세요.
    input_audio_path = "data/4minute.wav"
    
    # 'data' 디렉토리 생성
    os.makedirs("data", exist_ok=True)
    
    main(input_audio_path)
