## ~12/13
1. prediction.py 수정
2. file_prediction.py 수정
3. augmentation 적용

## ~12/12
1. Prerocessingdata.py / zero-padding 추가 및 변경</br>
2. training.py / 특징추출별 모델 변경 및 파라미터 변경, Learningrate 추가</br>
3. VS.py / 평균값 이미지 삭제 및 random 함수 추가</br>
4. 지표 결과 Spectrogram > Mel-S > MFCC > Waveform</br>
 4-1. 12/8 코멘트와 같이 절대지표 아님. 또한 모델 구조 및 파라미터 변경으로 매우 큰 폭의 변화 확인</br>
-. 학습조건 etc(1022ea), gaesaekki(970ea), shibal(940ea)</br>
-. to do list / 코드 수정에 따른 prediction.py 수정, file_prediction.py 적용을 통한 모델별 테스트 결과값 출력, 데이터셋 변경(현재 aug없음)</br>
-. saved_model, pkl 파일 수정 https://drive.google.com/drive/folders/1Z-TCXeJrZUE_m3mRfbdMlpe0FXBPRMH6


## 12/8
1. 본 프로젝트에 알맞는 음성특징 추출 방식 변경을 위한 특징추출방식 확장(旣 MFCC 및 추가 Mel-S, Spectrogram, Waveform)</br>
2. 추출 방식별 통상적으로 사용되는 모델(1D CNN, CNN, MLP 등)을 사용했으나 모델과 세부 param 변경 테스트 진행 要<br>
3. 팀원 공유 및 번거로운 작업을 피하고자 특징추출별 별도 py파일 생성</br>
4. 모든 학습환경 획일화(epoch, es patience, mc bestmodel, learning late 추가적용 및 획일화)</br>
5. confusion matrix 값 및 score evaluate 저장(Figure, score)</br>
6. 지표 결과 Mel-s >= MFCC > Waveform > Spectrogram 순</br>
 6-1. 라고 생각했으나 모델은 traning 진행할때마다 결과값이 큰 폭으로 변동됨을 확인했으며 6의 값이 절대값이라 전혀 볼 수 없음</br>
-. git main/simple_audio_edit/file_prediction.py 파일 적용 준비</br>
7. VS_특징추출별 시각화 파일 생성
8. 특징추출별 best_model과 pkl(csv대체) 용량이 매우 크므로 drive에 업로드 https://drive.google.com/drive/folders/1Z-TCXeJrZUE_m3mRfbdMlpe0FXBPRMH6

## 12/1
-. requirements 작성</br>
-. 학습에 사용한 dataset (https://drive.google.com/drive/folders/1EMoV-uc0N9bZXCYz_9vLNYepGSnpZSmc)</br>
-. 욕설 단어 학습환경 설명 (메인폴더 하위 myproject 폴더가 욕설 관련 폴더이니 myproject 하위에서 작업할 것)</br>
1. requirements.txt 확인 후 설치 (pip install -r requirements.txt)</br>
2. 직접 학습하려면  dataset 다운로드 및 개인 녹음본 myproject/dataset/gaesaekki, etc, shibal 3개 폴더 위치</br>
2-1. PreprocessingData.py → training.py → prediction.py 순 실행 후 테스트</br>
3. fileinput_prediction.py 를 통한 파일 predict</br>
4. SpecAugment_with_Noise.py 코드 수정
5. PitchAugment.py 파일 생성
6. requirements.txt 수정

## 11/30
-. myproject 폴더 생성</br>
-. 하위 PreprocessingData.py, training.py, prediction.py, SpecAugment_with_Noise.py 수정 및 재생성

## CNN, LSTM, CNN+LSTM
-. 학습 사용 데이터는 용량관계상 별도 보관  

## 학습 환경
-. down ~ yes 총 8개 라벨 원본 음성 데이터 라벨당 1천개</br>
-. 원본 음성 데이터 Augmentation(SpecAugment+Noise(1천개), pitch_shifted(300개), time_strecthed(300개)) 음성 데이터 총 1,600개  

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
