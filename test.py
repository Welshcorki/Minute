
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from scipy.io import wavfile
import os

def preprocess_audio(input_path, output_path):
    """
    음성 파일을 불러와 노이즈 제거 및 정규화를 수행하고 결과를 저장합니다.

    Args:
        input_path (str): 원본 음성 파일 경로
        output_path (str): 처리된 음성 파일을 저장할 경로
    """
    try:
        # 1. 음성 데이터 로드
        print(f"'{input_path}'에서 음성 파일을 로드합니다...")
        audio = AudioSegment.from_wav(input_path)
        print("로드 완료.")
        print(f"  - 길이: {len(audio) / 1000:.2f}초")
        print(f"  - 채널: {audio.channels}")
        print(f"  - 샘플링 레이트: {audio.frame_rate}Hz")
        print(f"  - 최대 진폭: {audio.max}")

        # 2. 노이즈 제거
        # pydub 오디오를 numpy 배열로 변환
        samples = np.array(audio.get_array_of_samples())
        
        print("\n노이즈 제거를 시작합니다...")
        # noisereduce를 위해 샘플링 레이트 필요
        rate = audio.frame_rate
        reduced_noise_samples = nr.reduce_noise(y=samples, sr=rate)
        print("노이즈 제거 완료.")

        # numpy 배열을 다시 pydub 오디오 세그먼트로 변환
        # 참고: 오디오의 sample_width와 channels를 유지해야 합니다.
        audio_no_noise = AudioSegment(
            reduced_noise_samples.tobytes(),
            frame_rate=rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # 3. 음량 정규화
        print("\n음량 정규화를 시작합니다...")
        normalized_audio = audio_no_noise.normalize()
        print("음량 정규화 완료.")
        print(f"  - 정규화 후 최대 진폭: {normalized_audio.max}")


        # 4. 처리된 오디오 파일로 저장
        print(f"\n처리된 오디오를 '{output_path}'에 저장합니다...")
        normalized_audio.export(output_path, format="wav")
        print("저장 완료.")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 사용자가 제공한 파일 경로
    # 중요: 'data/4minute.wav' 파일이 실제로 이 경로에 있어야 합니다.
    input_audio_path = os.path.join("data", "4minute.wav")
    output_audio_path = os.path.join("data", "4minute_processed.wav")
    
    # 'data' 디렉토리가 없으면 생성
    if not os.path.exists("data"):
        os.makedirs("data")
        
    print("--- 오디오 전처리 및 디버깅 시작 ---")
    preprocess_audio(input_audio_path, output_audio_path)
    print("\n--- 모든 작업 완료 ---")
