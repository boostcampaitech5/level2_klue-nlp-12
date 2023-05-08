import os
import torch
import sklearn
import numpy as np
import pandas as pd
import pickle as pickle
from metric import *
from load_data import *
from args import parse_arguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from transformers.integrations import WandbCallback
import wandb
from wandb import AlertLevel


def train(config):
    # load model and tokenizer
    MODEL_NAME = config.arch["type"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. load dataset
    # 2. preprocess dataset
    # 3. tokenize dataset
    revision = config.dataloader["revision"]

    train_dataset, train_raw_label = load_train_dataset("train[:80%]", revision, tokenizer)
    dev_dataset, dev_raw_label = load_train_dataset("train[80%:]", revision, tokenizer)  # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_raw_label)
    dev_label = label_to_num(dev_raw_label)

    # 4. make Dataset object
    RE_train_dataset = RE_Dataset(train_dataset, train_label)
    RE_dev_dataset = RE_Dataset(dev_dataset, dev_label)

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
    print(model.config)

    model.parameters
    model.to(device)

    # 6. training arguments 설정
    ## 사용한 option 외에도 다양한 option들이 있습니다.
    ## https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=config.trainer["output_dir"],  # output directory
        save_total_limit=config.trainer["save_total_limit"],  # number of total save model
        save_steps=config.trainer["save_steps"],  # model saving step
        num_train_epochs=config.trainer["epochs"],  # total number of training epochs
        learning_rate=config.optimizer["lr"],  # learning_rate
        per_device_train_batch_size=config.dataloader["batch_size"],  # batch size per device during training
        per_device_eval_batch_size=config.dataloader["batch_size"],  # batch size for evaluation
        warmup_steps=config.lr_scheduler["warmup_steps"],  # number of warmup steps for learning rate scheduler
        weight_decay=config.optimizer["weight_decay"],  # strength of weight decay
        logging_dir=config.trainer["logging_dir"],  # directory for storing logs
        logging_steps=config.trainer["logging_steps"],  # log saving step.
        evaluation_strategy=config.trainer["evaluation_strategy"],  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=config.trainer["evaluation_steps"],  # evaluation step
        load_best_model_at_end=True,
        report_to=("wandb" if config.use_wandb else "none"),  # integrations to report the results and logs to
    )

    # 7. trainer 설정
    # 8. evaluate 함수 설정
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=([WandbCallback()] if config.use_wandb else []),
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
    config = parse_arguments()

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
