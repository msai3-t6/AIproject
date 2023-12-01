## 12/1
-. requirements 작성</br>
-. 학습에 사용한 dataset (https://drive.google.com/drive/folders/1EMoV-uc0N9bZXCYz_9vLNYepGSnpZSmc)</br>
-. 욕설 단어 학습환경 설명</br>
1. requirements.txt 확인 후 설치</br>
2. dataset 다운로드 및 myproject/dataset/gaesaekki, etc, shibal 3개 폴더 위치</br>
3. 데이터셋 추가 등 수정 시 PreprocessingData.py → training.py → prediction.py 순 실행 후 테스트

## 11/30
-. myproject 폴더 생성</br>
-. 하위 PreprocessingData.py, training.py, prediction.py, SpecAugment_with_Noise.py 수정 및 재생성

## CNN, LSTM, CNN+LSTM
-. 학습 사용 데이터는 용량관계상 별도 보관  

## 학습 환경
-. down ~ yes 총 8개 라벨 원본 음성 데이터 라벨당 1천개</br>
-. 원본 음성 데이터 Augmentation(SpecAugment+Noise(1천개), pitch_shifted(300개), time_strecthed(300개) 음성 데이터 총 1,600개  

## 학습 조건
-. 기존 Hello-detection CNN 개선된 CNN (지이현, 황지운 수정)</br>
-. CNN to LSTM</br>
-. CNN + LSTM 

## Local PC 테스트 방법
-. Prediction.py load_model 코드를 mixed_LSTM, modified_cnn, switch_LSTM.h5 변경 후 실행(각 모델명 best_model.h5 파일도 사용 가능)</br>
-. 라벨 단어 마이크 사용으로 테스트

## 결과값
-. Figure 이미지 파일 3종, result 이미지 파일 3종으로 확인.

## 의견
-. LSTM은 장기적인 의존성 학습하는데 강력하고 복잡한 시계열 데이터, 자연어 처리 등에 효과적이므로 현재 우리가 진행하는 프로젝트는 사실상 1개 음절 단어로 이루어져 있기 때문에 효과를 보지 못하거나 오히려 성능이 저하됨을 확인.</br>
-. 다양한 음성을 학습하기 위해 augmentation 작업이 필수적인데, 우리 모델에 어떤 augmentation을 적용하며 수치를 어떻게 설정해야할지 고민(자원 낭비(컴퓨터 리소스, 시간))이 필요해보임.</br>
