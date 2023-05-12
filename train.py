import sys
import pickle as pickle

import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from transformers.integrations import WandbCallback
from wandb import AlertLevel

from loss import *
from utils import *
from metric import *
from load_data import *
from args import parse_arguments


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs= False):
        device= torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        labels= inputs.pop('labels')

        # forward pass
        outputs= model(**inputs)
        
        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]
            
        # compute custom loss
        # custom loss function, 아래에서 이름을 바꾸면 다른 loss 도 사용가능
        # 'lovasz_loss', 'focal_loss', 'smooth_L1_loss', 'default'
        custom_loss = change_loss_function('default').to(device)
        loss = custom_loss(outputs['logits'], labels)    
        return (loss, outputs) if return_outputs else loss


def train(config):
    seed_everything(config.seed)
    
    # load model and tokenizer
    MODEL_NAME = config.arch["type"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. load dataset
    # 2. preprocess dataset
    # 3. tokenize dataset
    revision = config.dataloader["revision"]
    input_format = config.dataloader.get("input_format")
    prompt = config.dataloader.get("prompt")

    train_dataset, train_raw_label = load_train_dataset(
        split = config.dataloader['train_split'],
        revision = revision,
        tokenizer = tokenizer,
        input_format = input_format,
        prompt = prompt
        )
    dev_dataset, dev_raw_label = load_train_dataset(
        split = config.dataloader['valid_split'],
        revision = revision,
        tokenizer = tokenizer,
        input_format = input_format,
        prompt = prompt
        )

    train_label = label_to_num(train_raw_label)
    dev_label = label_to_num(dev_raw_label)

    # 4. make Dataset object
    re_train_dataset = RE_Dataset(train_dataset, train_label)
    re_dev_dataset = RE_Dataset(dev_dataset, dev_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 5. import model
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    # import model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    print(model.config)

    model.parameters
    model.to(device)

    # 6. training arguments 설정
    ## 사용한 option 외에도 다양한 option들이 있습니다.
    ## https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        # 기본 설정
        output_dir=config.trainer["output_dir"],  # 모델 저장 디렉토리
        report_to=("wandb" if config.use_wandb else "none"),  # wandb 사용 여부
        fp16=True,

        # 학습 설정
        num_train_epochs=config.trainer["epochs"],  # 전체 훈련 epoch 수
        learning_rate=config.optimizer["lr"],  # learning rate
        weight_decay=config.optimizer["weight_decay"],  # weight decay
        adam_beta2=config.optimizer["adam_beta2"],  # AdamW 옵티마이저의 beta2 하이퍼파라미터

        # 배치 사이즈 설정
        per_device_train_batch_size=config.dataloader["batch_size"],  # 훈련 중 장치 당 batch size
        per_device_eval_batch_size=config.dataloader["batch_size"],  # 평가 중 장치 당 batch size

        # 스케줄링 설정
        warmup_ratio=config.lr_scheduler['warmup_ratio'],  # learning rate scheduler의 warmup 비율
        # warmup_steps=config.lr_scheduler["warmup_steps"],  # number of warmup steps for learning rate scheduler

        # 로깅 설정
        logging_dir=config.trainer["logging_dir"],  # 로그 저장 디렉토리
        logging_steps=config.trainer["logging_steps"],  # 로그 저장 스텝

        # 모델 저장 설정
        save_total_limit=config.trainer["save_total_limit"],  # 전체 저장 모델 수 제한
        save_steps=config.trainer["save_steps"],  # 모델 저장 스텝
        save_strategy=config.trainer["save_strategy"],

        # 평가 설정
        evaluation_strategy=config.trainer["evaluation_strategy"],  # 훈련 중 평가 전략
        eval_steps=config.trainer["evaluation_steps"],  # 평가 스텝
        load_best_model_at_end=True,
    )

    # 7. trainer 설정
    # 8. evaluate 함수 설정
    trainer = CustomTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=re_train_dataset,  # training dataset
        eval_dataset=re_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        # callbacks=([WandbCallback()] if config.use_wandb else []),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.trainer["early_stop"])],
    )

    # 9. train model
    # 10. evaluate model
    trainer.train()
    model.save_pretrained("./best_model")

    wandb.alert(title="finished", level=AlertLevel.INFO, text=f"{run_name}")


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def main(config):
    train(config)


if __name__ == "__main__":
    
    try:
        config_path = sys.argv[1]
    except:
        config_path = './config.yaml'
        
    config = parse_arguments(config_path)

    if config.use_wandb:
        run_name = "{}_{}_{}_{}_{}".format(
            config.arch["type"],
            config.dataloader["batch_size"],
            config.trainer["epochs"],
            config.optimizer["lr"],
            config.loss["type"],
        )

        wandb.init(
            entity=config.wandb["entity"],
            project=config.wandb["project_name"],
            name=run_name,
            config=config,
        )

        wandb.alert(title="start", level=AlertLevel.INFO, text=f"{run_name}")

    main(config)