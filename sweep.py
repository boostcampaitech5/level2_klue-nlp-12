import sys
import pickle as pickle
import pytz
from datetime import datetime
import wandb

import torch
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from args import *
from load_data import *
from model import *
from metric import *
from trainer import *
from utils import *

from typing import Any


def main(config) -> None:

    def sweep_train(config: Any = config) -> None:
        
        print(config.wandb)
        return

        wandb.init(
            entity=config.wandb['entity'],
            project=config.wandb['project_name'],
            #name=run_name,
            #config=config,
        )   

        sweep_config = wandb.config

        seed_everything(config.seed)

        # load model and tokenizer
        model_name = config.model['name']
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 1. load dataset
        # 2. preprocess dataset
        # 3. tokenize dataset
        revision = config.dataloader['revision']
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
        dev_dataset, dev_raw_label = load_train_dataset(
            split=config.dataloader['valid_split'],
            revision=revision,
            tokenizer=tokenizer,
            input_format=input_format,
            prompt=prompt,
            type_transform=type_transform,
        )

        train_label = label_to_num(train_raw_label)
        dev_label = label_to_num(dev_raw_label)

        # 4. make Dataset object
        re_train_dataset = REDataset(train_dataset, train_label)
        re_dev_dataset = REDataset(dev_dataset, dev_label)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        # 5. import model
        # setting model hyperparameter
        model_module = __import__('model', fromlist=[config.model['variant']])
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
            num_train_epochs=sweep_config['epochs'],  # 전체 훈련 epoch 수
            learning_rate=sweep_config['lr'],  # learning rate
            weight_decay=config.optimizer['weight_decay'],  # weight decay
            adam_beta2=config.optimizer['adam_beta2'],  # AdamW 옵티마이저의 beta2 하이퍼파라미터

            # 배치 사이즈 설정
            per_device_train_batch_size=sweep_config['batch_size'],  # 훈련 중 장치 당 batch size
            per_device_eval_batch_size=sweep_config['batch_size'],  # 평가 중 장치 당 batch size

            # 스케줄링 설정
            warmup_ratio=config.lr_scheduler['warmup_ratio'],  # learning rate scheduler의 warmup 비율
            # warmup_steps=config.lr_scheduler['warmup_steps'],  # number of warmup steps for learning rate scheduler

            # 로깅 설정
            logging_dir=config.trainer['logging_dir'],  # 로그 저장 디렉토리
            logging_steps=config.trainer['logging_steps'],  # 로그 저장 스텝

            # 모델 저장 설정
            save_total_limit=config.trainer['save_total_limit'],  # 전체 저장 모델 수 제한
            save_steps=config.trainer['save_steps'],  # 모델 저장 스텝
            save_strategy=config.trainer['save_strategy'],

            # 평가 설정
            evaluation_strategy=config.trainer['evaluation_strategy'],  # 훈련 중 평가 전략
            eval_steps=config.trainer['evaluation_steps'],  # 평가 스텝
            load_best_model_at_end=True,
        )

        # 7. trainer 설정
        # 8. evaluate 함수 설정
        trainer = RETrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=re_train_dataset,  # training dataset
            eval_dataset=re_dev_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            # callbacks=([WandbCallback()] if config.use_wandb else []),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config.trainer['early_stop'])],
            loss_cfg=config.loss,
        )

        # 9. train model
        trainer.train()
        # 10. save model
        trainer.save_model(config.trainer['model_dir'])

    
    sweep_id = wandb.sweep(
        sweep=config.sweep_config
    )

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=config.wandb['sweep_count']
    )


if __name__ == '__main__':
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = './config.yaml'
    config = parse_arguments(config_path)

    now = datetime.now(pytz.timezone('Asia/Seoul'))
    run_name = f'{config.run_name}_{now.strftime("%d-%H-%M")}'

    main(config)