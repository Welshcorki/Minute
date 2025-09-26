# -*- coding: utf-8 -*-
import os
import json
import logging

def save_results(diarization_result, corrected_diarization, summary, original_filename, results_dir, meeting_topic, keywords):
    """
    변환된 텍스트와 요약, 교정된 내용을 파일로 저장합니다.
    """
    os.makedirs(results_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]

    # 1. 원본 STT 결과 (TXT)
    txt_filename = os.path.join(results_dir, f"stt_{base_filename}.txt")
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            for segment in diarization_result:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
        logging.info(f"STT 결과를 '{txt_filename}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({txt_filename}): {e}")

    # 2. LLM 교정 결과 (TXT)
    corrected_txt_filename = os.path.join(results_dir, f"corrected_{base_filename}.txt")
    try:
        with open(corrected_txt_filename, "w", encoding="utf-8") as f:
            for segment in corrected_diarization:
                f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['speaker']}: {segment['text']}\n")
        logging.info(f"LLM 교정 결과를 '{corrected_txt_filename}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({corrected_txt_filename}): {e}")

    # 3. 회의 요약 결과 (MD)
    summary_filename = os.path.join(results_dir, f"summary_{base_filename}.md")
    try:
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(f"# 회의 요약: {meeting_topic}\n\n")
            f.write("## 주요 내용\n")
            f.write(summary + "\n\n")
            f.write("## 전체 대화 내용 (교정본)\n")
            for segment in corrected_diarization:
                f.write(f"- **{segment['speaker']}**: {segment['text']}\n")
        logging.info(f"회의 요약 및 전체 대화 내용을 '{summary_filename}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({summary_filename}): {e}")

    # 4. JSON 결과 (원본, 교정본 포함)
    json_filename = os.path.join(results_dir, f"diarization_{base_filename}.json")
    combined_results = {
        "meeting_topic": meeting_topic,
        "keywords": keywords,
        "original_transcript": diarization_result,
        "corrected_transcript": corrected_diarization,
        "summary": summary
    }
    try:
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=4)
        logging.info(f"모든 결과를 '{json_filename}'에 저장했습니다.")
    except IOError as e:
        logging.error(f"파일 저장 중 오류 발생 ({json_filename}): {e}")
