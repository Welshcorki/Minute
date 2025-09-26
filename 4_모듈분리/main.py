# -*- coding: utf-8 -*-
import sys
import time
import logging
from openai import OpenAI
from pydub import AudioSegment

# 모듈 임포트
from test05.config import MEETING_TOPIC, KEYWORDS, AUDIO_FILE_PATH, RESULTS_DIR, TEMP_SEGMENT_FILENAME
from test05.api_keys import load_api_keys
from test05.diarization import diarize_audio
from test05.transcription import transcribe_segment
from test05.llm_processing import correct_text_with_llm, summarize_text
from test05.save_results import save_results

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    메인 실행 함수
    """
    # 1. API 키 로드
    openai_api_key, pyannote_token = load_api_keys()
    if not openai_api_key or not pyannote_token:
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)

    # 2. 화자 분리
    diarization = diarize_audio(AUDIO_FILE_PATH, pyannote_token)
    if not diarization:
        sys.exit(1)

    try:
        audio = AudioSegment.from_wav(AUDIO_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"오디오 파일을 찾을 수 없습니다: {AUDIO_FILE_PATH}")
        sys.exit(1)

    diarization_result = []
    logging.info("각 화자 세그먼트의 음성 인식을 시작합니다...")
    stt_start_time = time.time()

    # STT 프롬프트 생성
    stt_prompt = f"이 대화는 '{MEETING_TOPIC}'에 관한 것입니다. 주요 용어는 다음과 같습니다: {', '.join(KEYWORDS)}."

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = turn.start * 1000
        end_ms = turn.end * 1000
        segment_audio = audio[start_ms:end_ms]
        
        text = transcribe_segment(client, segment_audio, TEMP_SEGMENT_FILENAME, stt_prompt)
        
        if text:
            diarization_result.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "text": text
            })
            print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}: {text}")

    stt_end_time = time.time()
    logging.info(f"음성 인식 완료. (총 처리 시간: {stt_end_time - stt_start_time:.2f}초)")

    if not diarization_result:
        logging.warning("음성 인식 결과가 없습니다. 프로그램을 종료합니다.")
        sys.exit(0)

    # 3. LLM을 이용한 전체 텍스트 교정
    full_transcript = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in diarization_result)
    corrected_full_transcript = correct_text_with_llm(client, full_transcript, MEETING_TOPIC, KEYWORDS)

    # 교정된 전체 텍스트를 다시 화자별로 분리
    corrected_lines = corrected_full_transcript.strip().split('\n')
    corrected_diarization_result = []
    for i, segment in enumerate(diarization_result):
        new_text = segment['text']
        if i < len(corrected_lines):
            parts = corrected_lines[i].split(':', 1)
            if len(parts) > 1:
                new_text = parts[1].strip()
        
        corrected_diarization_result.append({
            "start": segment['start'],
            "end": segment['end'],
            "speaker": segment['speaker'],
            "text": new_text
        })

    # 4. 전체 대화 내용 요약
    full_corrected_transcript_for_summary = "\n".join(seg['text'] for seg in corrected_diarization_result)
    summary = summarize_text(client, full_corrected_transcript_for_summary, MEETING_TOPIC, KEYWORDS)

    # 5. 결과 저장
    save_results(
        diarization_result,
        corrected_diarization_result,
        summary,
        AUDIO_FILE_PATH,
        RESULTS_DIR,
        MEETING_TOPIC,
        KEYWORDS
    )

if __name__ == "__main__":
    main()
