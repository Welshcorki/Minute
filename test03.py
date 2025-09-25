# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_api_keys():
    """
    .env 파일에서 API 키를 로드하고 설정합니다.
    """
    try:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        pyannote_token = os.getenv("PYANNOTE_TOKEN")
        if not openai_api_key:
            logging.error("OPENAI_API_KEY가 .env 파일에 없습니다.")
        if not pyannote_token:
            logging.error("PYANNOTE_TOKEN이 .env 파일에 없습니다.")
        return openai_api_key, pyannote_token
    except Exception as e:
        logging.error(f"API 키 로드 중 오류 발생: {e}")
        return None, None

def diarize_audio(audio_path, token):
    """
    pyannote.audio를 사용하여 오디오 파일의 화자를 분리합니다.
    """
    if not os.path.exists(audio_path):
        logging.error(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        return None
    
    logging.info("화자 분리를 시작합니다...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diarization = pipeline(audio_path)
        logging.info("화자 분리 완료.")
        return diarization
    except Exception as e:
        logging.error(f"화자 분리 중 오류 발생: {e}")
        return None

def transcribe_segment(client, audio_segment, segment_path):
    """
    Whisper API를 사용하여 오디오 세그먼트를 텍스트로 변환합니다.
    """
    try:
        audio_segment.export(segment_path, format="wav")
        with open(segment_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        logging.error(f"Whisper API 호출 중 오류 발생: {e}")
        return ""
    finally:
        if os.path.exists(segment_path):
            os.remove(segment_path)

def save_results(diarization_result, original_filename):
    """
    변환된 텍스트를 다양한 형식으로 파일로 저장합니다.
    """
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    
    # TXT 파일로 저장
    txt_filename = f"diarization_{base_filename}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        for segment in diarization_result:
            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
    logging.info(f"변환된 텍스트를 '{txt_filename}' 파일에 저장했습니다.")

    # JSON 파일로 저장
    json_filename = f"diarization_{base_filename}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(diarization_result, f, ensure_ascii=False, indent=4)
    logging.info(f"변환된 텍스트를 '{json_filename}' 파일에 저장했습니다.")


def main():
    """
    메인 실행 함수
    """
    # 1. API 키 로드
    openai_api_key, pyannote_token = load_api_keys()
    if not openai_api_key or not pyannote_token:
        sys.exit(1)

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=openai_api_key)

    # 2. 오디오 파일 경로를 커맨드 라인 인자로부터 받음
    if len(sys.argv) < 2:
        logging.error("사용법: python test03.py <오디오 파일 경로>")
        logging.info("예시: python test03.py data/4minute.wav")
        sys.exit(1)
        
    audio_path = sys.argv[1]

    # 3. 화자 분리
    diarization = diarize_audio(audio_path, pyannote_token)

    if diarization:
        audio = AudioSegment.from_wav(audio_path)
        diarization_result = []
        
        logging.info("각 화자 세그먼트의 음성 인식을 시작합니다...")
        start_time = time.time()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = turn.start * 1000
            end_ms = turn.end * 1000
            segment_audio = audio[start_ms:end_ms]
            
            segment_filename = f"temp_segment.wav"
            text = transcribe_segment(client, segment_audio, segment_filename)
            
            if text:
                diarization_result.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "text": text
                })
                print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}: {text}")

        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(f"모든 세그먼트 음성 인식 완료. (총 처리 시간: {processing_time:.2f}초)")

        # 4. 결과 저장
        if diarization_result:
            save_results(diarization_result, audio_path)

if __name__ == "__main__":
    main()
