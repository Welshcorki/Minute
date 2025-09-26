# -*- coding: utf-8 -*-
import os
import logging

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
