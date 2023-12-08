## Base Model 
[Tensorflow 예제 Simple Audio ](https://www.tensorflow.org/tutorials/audio/simple_audio?hl=ko) 를 베이스 모델로 사용하여 수정 

## 수정내용 

데이터셋
- 개새끼 2880개 
- 씨발 3756개 
- 기타 2950개

  ㄴ 배경 사운드 350개

  ㄴ 지운님 목소리 추가 100개

  ㄴ 기타 2500개 (AI HUB 음성 데이터) 

* 기본음성 + Spec Augment (지운님 어그멘테이션 / 노이즈 추가) + 추가 Augmentation (배경음추가 / 피치 / 스트레치) 
* 개새끼는 지운님 음성 100개 삭제 + 도연님 음성이 없어서 CLASS 의 숫자 차이 발생
* 씨발 개수를 줄이면 씨발에 대한 정확도가 낮아져서 그대로 뒀어요 

## 실행방법 
1. 데이터 셋 아래 폴더 구조로 저장 ( dataset ㄴgaesaekki ㄴshibal) 
2. simple_audio.ipynb 실행하여 학습 및 모델 저장

   - DATASET_PATH = 'C:\\Users\\wise_\\pj2\\simple_audio_20231128\\16bit_datasetset' 라고 되어있는 부분을 개인 폴더 경로로 지정 필요!

   - 상대경로로 지정하면 계속 에러나서 여기만 절대경로로 되어있습니다. 

3. main.py 실행하면 음성 녹음 및 예측 진행 가능

## 코드 문제점 및 개선 필요 사항 
- 일부 단어를 잘못 예측하는 문제가 있음

- 기타음성 하기 데이터셋에서 랜덤 추출 :
-    백그라운드 사운드 (https://github.com/karolpiczak/ESC-50)
-    AI 허브 화자 인식용 음성 데이터(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=537) 


모델 버전 업데이트 이력


|폴더명|데이터셋|이슈사항|Accuracy|
|------|---|---|---|
|saved_model_3class_v2|씨발/개새끼 각 300개 + 기타 250개 |others 를 잘 감지하지 못하는 문제|59.2%|
|saved_model_3class_hundred_thousand|씨발/개새끼 AUG 각 600개 + 기타 약 13만개 |  class 데이터 불균형으로 인해 모든 것을 others 로 감지|38.3%|
|saved_model_3class_1900|씨발/개새끼 AUG 각 600개 + 기타 약 700개 | others 를 잘 감지하지 못함 |65.0%|
|saved_model_3class_3000|씨발/개새끼 AUG 각 600개 + 기타 약 1200개 | 개새끼 를 잘 감지하지 못함 |55.0%|
|saved_model_3class_4200|씨발/개새끼 AUG 각 1400개 + 기타 약 1400개 | 개새끼 & 씨발 둘 다 잘 감지하지 못함 |62.5%|
|saved_model_3class_3800|씨발/개새끼 AUG 각 1200개(정직님 음성 제외) + 기타 약 1400개 | 씨발은 개선되었지만 개새끼는 낮은 정확도 |65.8%|
|saved_model_3class_3200|씨발 1200개 + 개새끼 1000개(지운님 음성 일부 삭제) + 기타 1000개 | 개새끼 를 잘 감지하지 못함 |67.5%|
|saved_model_3class_3100|씨발 1200개 + 개새끼 900개(지운님 음성 100개만 사용) + 기타 1000개 | 개새끼 아직 잘 감지 못하지만 이전 대비 개선 |73.3%|
|saved_model_7600_add_dropout_one05| 확정된 데이터 셋 사용  | 개새끼 아직 잘 감지 못하지만 이전 대비 개선 |72.5%|



## 라벨별 세부 Accuracy Percentage 
![image](https://github.com/ractactia/AIproject/assets/137852127/d56bd7e1-1f80-4056-9d45-167fe3a37bc1)



