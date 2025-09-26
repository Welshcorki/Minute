# -*- coding: utf-8 -*-
import os
import logging
from pyannote.audio import Pipeline

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
