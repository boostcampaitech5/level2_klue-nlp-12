import os
import torch
import pandas as pd
import pickle as pickle
from datasets import load_dataset


def load_train_dataset(split, revision, tokenizer):
    """train dataset을 불러온 후, tokenizing 합니다."""
    dataset = load_dataset(
        "Smoked-Salmon-s/RE_Competition",
        split=split,
        column_names=[
            "id",
            "sentence",
            "subject_entity",
            "object_entity",
            "label",
            "source",
        ],
        revision=revision,
    )
    pd_dataset = (
        dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({"id": "int64"})
    )
    train_dataset = preprocessing_dataset(pd_dataset)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    train_label = pd_dataset["label"].values

    return tokenized_train, train_label


def load_test_dataset(split, revision, tokenizer):
    """test dataset을 불러온 후, tokenizing 합니다."""
    dataset = load_dataset(
        "Smoked-Salmon-s/RE_Competition",
        split=split,
        column_names=[
            "id",
            "sentence",
            "subject_entity",
            "object_entity",
            "label",
            "source",
        ],
        revision=revision,
    )
    pd_dataset = (
        dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({"id": "int64"})
    )
    test_dataset = preprocessing_dataset(pd_dataset)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    if split == "test":
        test_label = list(map(int, pd_dataset["label"].values))
    else:
        test_label = pd_dataset["label"].values

    return test_dataset["id"], tokenized_test, test_label


def preprocessing_dataset(dataset):
    """subject_entity column과 object_entity column을 리스트 형태로 변환합니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]
        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )

    return out_dataset


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    return tokenized_sentences


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
