## Base Model 
[Tensorflow 예제 Simple Audio ](https://www.tensorflow.org/tutorials/audio/simple_audio?hl=ko) 를 베이스 모델로 사용하여 수정 

## 수정내용 

12/1  
- 3명 음성을 학습하도록 수정


## 실행방법 
1. 데이터 셋 아래 폴더 구조로 저장 ( dataset ㄴgaesaekki ㄴshibal) 
2. simple_audio.ipynb 실행하여 학습 및 모델 저장

   - DATASET_PATH = 'C:\\Users\\wise_\\pj2\\simple_audio_20231128\\16bit_datasetset' 라고 되어있는 부분을 개인 폴더 경로로 지정 필요!

   - 상대경로로 지정하면 계속 에러나서 여기만 절대경로로 되어있습니다. 

3. main.py 실행하면 음성 녹음 및 예측 진행 가능

## 코드 문제점 및 개선 필요 사항 
- 일부 단어를 잘못 예측하는 문제가 있음 


모델 버전 업데이트 이력


|폴더명|데이터셋|이슈사항|
|------|---|---|
|saved_model_3class_v2|씨발/개새끼 300개 + 기타 250개 |others 를 잘 감지하지 못하는 문제|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
