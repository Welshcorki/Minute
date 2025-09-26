# -*- coding: utf-8 -*-
import os
import logging
from dotenv import load_dotenv

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
            return None, None
        if not pyannote_token:
            logging.error("PYANNOTE_TOKEN이 .env 파일에 없습니다.")
            return None, None
        return openai_api_key, pyannote_token
    except Exception as e:
        logging.error(f"API 키 로드 중 오류 발생: {e}")
        return None, None
