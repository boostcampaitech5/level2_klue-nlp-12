# 문장 내 개체간 관계 추출 Relation Extraction
boostcamp AI Tech 5 NLP 트랙 레벨2 프로젝트  
문장 내의 두 단어(Entity)에 대한 속성과 관계를 예측하는 인공지능 만들기


## 일정 Schedule
프로젝트 전체 기간 : 5월 3일 (화) ~ 5월 18일 (목) 19:00


## 팀원 Team Members
|문지혜|박경택|박지은|송인서|윤지환|
|:---:|:---:|:---:|:---:|:---:|
|<img src="https://avatars.githubusercontent.com/u/85336141?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97149910?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97666193?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/41552919?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/37128004?v=4" width="120" height="120">|
|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:munjh1121@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:afterthougt@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:imhappyhill@gmail.com)](mailto:imhappyhill@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:songinseo0910@gmail.com)](mailto:songinseo0910@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yjh091500@naver.com)](mailto:yjh091500@naver.com)|
|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/jihye-moon)](https://github.com/jihye-moon)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/afterthougt)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/iamzieun)](https://github.com/iamzieun)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/fortunetiger)](https://github.com/fortunetiger)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/ohilikeit)](https://github.com/ohilikeit)|


## 프로젝트 보고서 및 발표 시각 자료 Report and Presentation
- [Wrap-Up Report](https://github.com/ohilikeit/level2_klue-nlp-12/blob/main/documents/wrap%20up%20report.pdf)  
- [Presentation](https://github.com/ohilikeit/level2_klue-nlp-12/blob/main/documents/presentation.pdf)


## 프로젝트 개요 Project Overview
<img src="https://github.com/boostcampaitech5/level2_klue-nlp-12/assets/97666193/0cf91f64-9302-4028-a3ea-43e8c35de3c8" width="1000" height="500"/>

## 레포지토리 구조 Repository Structure
```bash
level2_klue-nlp-12/
├── eda                         // eda 및 사후 분석용 함수
│   └── post_eda.py
│
├── load_data                   // 데이터 불러오기 관련 폴더
│   ├── dict_label_to_num.pkl   // 레이블을 숫자로 변환하기 위한 사전 파일
│   ├── dict_num_to_label.pkl   // 숫자를 레이블로 변환하기 위한 사전 파일
│   └── load_data.py            // 데이터 불러오기 및 전처리 관련 함수
│
├── model                       // 모델, 손실 함수, 평가 지표
│   ├── loss.py                 // 손실 함수
│   ├── metric.py               // 평가 지표
│   └── model.py                // 모델 아키텍쳐
│
├── trainer                     // 학습 관련 폴더
│   └──trainer.py
│
├── utils                       // 유틸리티 함수
│   ├── args.py                 // 프로그램 실행 시 전달되는 인자들을 처리하기 위한 파일
│   └── utils.py
│
├── documents                   // 보고서 및 발표자료
│   └── wrap up report.pdf
│   └── presentation.pdf
│
├── requirements.txt            // 프로젝트에 필요한 라이브러리들을 명시
│
├── train.py                    // 모델 학습 시작을 위한 메인 스크립트
├── full_train.py               // 전체 데이터로의 모델 학습 시작을 위한 메인 스크립트
├── inference.py                // 학습된 모델의 평가 및 추론을 위한 스크립트
├── sweep.py                    // sweep 동작을 위한 스크립트
│
├── config.yaml                 // 모델 학습 설정 관리를 위한 YAML
├── config_full.yaml            // 전체 데이터로의 모델 학습 설정 관리를 위한 YAML
│
├── run.sh                      // 실험 자동화를 위한 쉘 스크립트
├── pyproject.toml              // Black 설정 파일 
│
└── README.md
```

## 데이터 Data
- train.csv: 총 32470개
- test_data.csv: 총 7765개


## 사용법 Usage
- 모델 학습 및 추론 위해서는, [Huggingface Datasets](https://huggingface.co/datasets/Smoked-Salmon-s/RE_Competition) 로그인 및 access token 인증이 선행되어야 합니다. 
- 환경 설치
```bash
pip install -r requirement.txt
```
- 학습
```bash
python train.py config.yaml
```
- 전체 데이터셋에 대한 학습
```bash
python full_train.py config_full.yaml
```
- 추론
```bash
python inference.py config.yaml
```
- 린팅
```bash
black .
```
