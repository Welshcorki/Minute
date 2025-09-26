# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import json
from openai import OpenAI
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

# --- 설정 ---
# 회의 주제 및 주요 키워드 정의
MEETING_TOPIC = "사내 소통 활성화"
KEYWORDS = ["소통", "뉴스레터", "워크숍", "게시판", "부서 협업"]

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

def transcribe_segment(client, audio_segment, segment_path, prompt):
    """
    Whisper API를 사용하여 오디오 세그먼트를 텍스트로 변환합니다.
    주제와 키워드를 프롬프트에 포함하여 정확도를 높입니다.
    """
    try:
        audio_segment.export(segment_path, format="wav")
        with open(segment_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt=prompt
            )
        return transcript.text
    except Exception as e:
        logging.error(f"Whisper API 호출 중 오류 발생: {e}")
        return ""
    finally:
        if os.path.exists(segment_path):
            os.remove(segment_path)

def correct_text_with_llm(client, text, topic, keywords):
    """
    LLM을 사용하여 텍스트를 교정합니다.
    """
    logging.info("LLM으로 텍스트 교정을 시작합니다...")
    try:
        prompt = f"""다음 텍스트는 '{topic}'에 대한 회의 내용입니다. 
        주요 키워드는 {', '.join(keywords)} 입니다. 
        문맥에 맞게 문장을 다듬고, 맞춤법 및 띄어쓰기를 수정해주세요. 
        특히, 키워드가 포함된 문장은 더 자연스럽게 만들어주세요.

        원본 텍스트:
        {text}

        교정된 텍스트:
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects and refines meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        corrected_text = response.choices[0].message.content.strip()
        logging.info("LLM 텍스트 교정 완료.")
        return corrected_text
    except Exception as e:
        logging.error(f"LLM 교정 중 오류 발생: {e}")
        return text # 교정 실패 시 원본 텍스트 반환

def save_results(diarization_result, corrected_diarization, summary, original_filename):
    """
    변환된 텍스트와 요약, 교정된 내용을 파일로 저장합니다.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]

    # 1. 원본 STT 결과 (TXT)
    txt_filename = os.path.join(results_dir, f"stt_{base_filename}.txt")
    with open(txt_filename, "w", encoding="utf-8") as f:
        for segment in diarization_result:
            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
    logging.info(f"STT 결과를 '{txt_filename}'에 저장했습니다.")

    # 2. LLM 교정 결과 (TXT)
    corrected_txt_filename = os.path.join(results_dir, f"corrected_{base_filename}.txt")
    with open(corrected_txt_filename, "w", encoding="utf-8") as f:
        for segment in corrected_diarization:
            f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
    logging.info(f"LLM 교정 결과를 '{corrected_txt_filename}'에 저장했습니다.")

    # 3. 회의 요약 결과 (MD)
    summary_filename = os.path.join(results_dir, f"summary_{base_filename}.md")
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write(f"# 회의 요약: {MEETING_TOPIC}\n\n")
        f.write("## 주요 내용\n")
        f.write(summary + "\n\n")
        f.write("## 전체 대화 내용 (교정본)\n")
        for segment in corrected_diarization:
            f.write(f"- **{segment['speaker']}**: {segment['text']}\n")
    logging.info(f"회의 요약 및 전체 대화 내용을 '{summary_filename}'에 저장했습니다.")

    # 4. JSON 결과 (원본, 교정본 포함)
    json_filename = os.path.join(results_dir, f"diarization_{base_filename}.json")
    combined_results = {
        "meeting_topic": MEETING_TOPIC,
        "keywords": KEYWORDS,
        "original_transcript": diarization_result,
        "corrected_transcript": corrected_diarization,
        "summary": summary
    }
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)
    logging.info(f"모든 결과를 '{json_filename}'에 저장했습니다.")


def summarize_text(client, text, topic, keywords):
    """
    LLM을 사용하여 전체 대화 내용을 요약합니다.
    """
    logging.info("LLM으로 회의 요약을 시작합니다...")
    try:
        prompt = f"""다음은 '{topic}'을 주제로 한 회의의 전체 대화 내용입니다.
        주요 키워드는 {', '.join(keywords)} 입니다.
        이 회의의 핵심 내용을 요약해주세요.

        전체 대화 내용:
        {text}

        회의 요약:
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        logging.info("LLM 회의 요약 완료.")
        return summary
    except Exception as e:
        logging.error(f"LLM 요약 중 오류 발생: {e}")
        return "요약 생성에 실패했습니다."

def main():
    """
    메인 실행 함수
    """
    audio_file_path = "C:/Users/SBA/github/Minute/data/20250923_script2.wav"

    # 1. API 키 로드
    openai_api_key, pyannote_token = load_api_keys()
    if not openai_api_key or not pyannote_token:
        sys.exit(1)

    client = OpenAI(api_key=openai_api_key)

    # 2. 화자 분리
    diarization = diarize_audio(audio_file_path, pyannote_token)
    if not diarization:
        sys.exit(1)

    audio = AudioSegment.from_wav(audio_file_path)
    diarization_result = []
    
    logging.info("각 화자 세그먼트의 음성 인식을 시작합니다...")
    stt_start_time = time.time()

    # STT 프롬프트 생성
    stt_prompt = f"이 대화는 '{MEETING_TOPIC}'에 관한 것입니다. 주요 용어는 다음과 같습니다: {', '.join(KEYWORDS)}."

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = turn.start * 1000
        end_ms = turn.end * 1000
        segment_audio = audio[start_ms:end_ms]
        
        segment_filename = f"temp_segment.wav"
        text = transcribe_segment(client, segment_audio, segment_filename, stt_prompt)
        
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

    # 3. LLM을 이용한 전체 텍스트 교정
    corrected_diarization_result = []
    full_transcript_for_correction = ""
    for segment in diarization_result:
        full_transcript_for_correction += f"{segment['speaker']}: {segment['text']}\n"

    corrected_full_transcript = correct_text_with_llm(client, full_transcript_for_correction, MEETING_TOPIC, KEYWORDS)
    
    # 교정된 전체 텍스트를 다시 화자별로 분리 (간단한 방법)
    # 이 부분은 더 정교한 방법으로 개선될 수 있습니다.
    corrected_lines = corrected_full_transcript.strip().split('\n')
    
    # 원본 diarization_result의 길이를 기준으로 교정된 텍스트를 할당합니다.
    # 이는 LLM이 출력 형식을 유지한다는 가정 하에 동작합니다.
    corrected_diarization_result = []
    for i, segment in enumerate(diarization_result):
        new_text = segment['text'] # 기본값은 원본 텍스트
        if i < len(corrected_lines):
            # "SPEAKER_ID: text" 형식에서 text 부분만 추출
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
    full_corrected_transcript = "\n".join(seg['text'] for seg in corrected_diarization_result)
    summary = summarize_text(client, full_corrected_transcript, MEETING_TOPIC, KEYWORDS)

    # 5. 결과 저장
    if diarization_result:
        save_results(diarization_result, corrected_diarization_result, summary, audio_file_path)

if __name__ == "__main__":
    main()
