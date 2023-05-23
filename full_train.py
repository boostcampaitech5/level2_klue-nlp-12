import sys
import pickle as pickle
import pytz
from datetime import datetime
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from utils.args import *
from load_data.load_data import *
from model.model import *
from model.metric import *
from trainer.trainer import *
from utils.utils import *


def train(config: Any) -> None:  
    """
    모델을 학습하는 함수, train.py와 다른 점은 평가 데이터셋 사용하지 않고 모든 데이터를 훈련에 활용함
    
    다음 프로세스를 수행:
        1. 데이터셋을 불러오고 전처리 및 토큰화
        2. 레이블을 숫자 형태로 변환
        3. 학습 및 개발 데이터셋에 대한 Dataset 객체를 생성
        4. 지정된 모델을 불러와 훈련 인자 설정
        5. 모델 학습 후 저장

    Args:
        config (dict): 모델 학습에 필요한 모든 구성 매개변수를 포함하는 딕셔너리
                       dataloader, model, optimizer, trainer 구성 포함

    Returns:
        None
    """
    seed_everything(config.seed)
    
    # load model and tokenizer
    model_name = config.model['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. load dataset
    # 2. preprocess dataset
    # 3. tokenize dataset
    revision = "69b6010fe9681567b98f9d3d3c70487079183d4b"
    input_format = config.dataloader.get('input_format')
    prompt = config.dataloader.get('prompt')
    type_transform = config.dataloader.get('type_transform')

    train_dataset, train_raw_label = load_train_dataset(
        split=config.dataloader['train_split'],
        revision=revision,
        tokenizer=tokenizer,
        input_format=input_format,
        prompt=prompt,
        type_transform=type_transform,
    )

    train_label = label_to_num(train_raw_label)

    # 4. make Dataset object
    re_train_dataset = REDataset(train_dataset, train_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 5. import model
    # setting model hyperparameter
    model_module = __import__('model.model', fromlist=[config.model['variant']])
    model_class = getattr(model_module, config.model['variant'])
    # Available customized classes:
    #   BaseREModel, BiLSTMREModel, BiGRUREModel
    model = model_class(config, len(tokenizer))

    print(model.model_config)

    model.parameters
    model.to(device)

    # 6. training arguments 설정
    ## 사용한 option 외에도 다양한 option들이 있습니다.
    ## https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        # 기본 설정
        output_dir=config.trainer['output_dir'],  # 모델 저장 디렉토리
        report_to=('wandb' if config.use_wandb else 'none'),  # wandb 사용 여부
        fp16=True,  # 16-bit floating point precision

        # 학습 설정
        num_train_epochs=config.trainer['epochs'],  # 전체 훈련 epoch 수
        learning_rate=config.optimizer['lr'],  # learning rate
        weight_decay=config.optimizer['weight_decay'],  # weight decay
        adam_beta2=config.optimizer['adam_beta2'],  # AdamW 옵티마이저의 beta2 하이퍼파라미터

        # 배치 사이즈 설정
        per_device_train_batch_size=config.dataloader['batch_size'],  # 훈련 중 장치 당 batch size

        # 스케줄링 설정
        warmup_ratio=config.lr_scheduler['warmup_ratio'],  # learning rate scheduler의 warmup 비율

        # 로깅 설정
        logging_dir=config.trainer['logging_dir'],  # 로그 저장 디렉토리
        logging_steps=config.trainer['logging_steps'],  # 로그 저장 스텝
        load_best_model_at_end=config.trainer['use_early_stop'],
    )

    # 7. trainer 설정
    # 8. evaluate 함수 설정
    trainer = RETrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=re_train_dataset,  # training dataset
        compute_metrics=compute_metrics,  # define metrics function
        loss_cfg=config.loss,
    )

    # 9. train model
    trainer.train()
    # 10. save model
    trainer.save_model(config.trainer['model_dir'])


def main() -> None:
    """
    config를 불러오고 학습 프로세스를 시작하는 메인 함수입니다.
    
    다음 프로세스를 수행:
        1. 제공된 YAML 파일에서 구성을 파싱하거나 기본 파일을 사용
        2. 제공된 구성으로 Weights & Biases (wandb) 실행 초기화
        3. train 함수를 호출하여 모델 훈련 프로세스를 시작
        4. 학습 완료 후 wandb에 완료 메세지 송출

    Args:
        None

    Returns:
        None
    """
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = './config.yaml'
    config = parse_arguments(config_path)

    now = datetime.now(pytz.timezone('Asia/Seoul'))
    run_name = f'{config.run_name}_{now.strftime("%d-%H-%M")}'

    init_wandb(config, run_name)
    train(config)
    alert_wandb(config, run_name, 'finished')


if __name__ == '__main__':
    main()