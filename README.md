# CNN, LSTM, CNN+LSTM
-. 학습 사용 데이터는 용량관계상 별도 보관  

# 학습 환경
down ~ yes 총 8개 라벨 원본 음성 데이터 라벨당 1천개
-. 원본 음성 데이터 Augmentation(SpecAugment+Noise(1천개), pitch_shifted(300개), time_strecthed(300개) 음성 데이터 총 1,600개  

# 학습 조건
-. 기존 Hello-detection CNN 개선된 CNN (지이현, 황지운 수정)</br>
-. CNN to LSTM</br>
-. CNN + LSTM 

# Local PC 테스트 방법
-. Prediction.py load_model 코드를 mixed_LSTM, modified_cnn, switch_LSTM.h5 변경 후 실행(각 모델명 best_model.h5 파일도 사용 가능)</br>
-. 라벨 단어 마이크 사용으로 테스트
