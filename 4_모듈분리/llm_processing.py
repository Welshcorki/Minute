# -*- coding: utf-8 -*-
import logging

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
