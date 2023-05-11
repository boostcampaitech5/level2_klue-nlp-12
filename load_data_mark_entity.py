import os
import re
import torch
import pandas as pd
import pickle as pickle
from tqdm import tqdm
from datasets import load_dataset


def load_train_dataset(split, revision, tokenizer, input_format):
    """train dataset을 불러온 후, tokenizing 합니다."""
    dataset = load_dataset(
        "Smoked-Salmon-s/RE_Competition",
        split=split,
        column_names=["id", "sentence", "subject_entity", "object_entity", "label", "source"],
        revision=revision,
    )
    pd_dataset = dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({"id": "int64"})
    train_dataset = preprocessing_dataset(pd_dataset, input_format)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, input_format)
    train_label = pd_dataset["label"].values

    return tokenized_train, train_label


def load_test_dataset(split, revision, tokenizer, input_format):
    """test dataset을 불러온 후, tokenizing 합니다."""
    dataset = load_dataset(
        "Smoked-Salmon-s/RE_Competition",
        split=split,
        column_names=["id", "sentence", "subject_entity", "object_entity", "label", "source"],
        revision=revision,
    )
    pd_dataset = dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({"id": "int64"})
    test_dataset = preprocessing_dataset(pd_dataset, input_format)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, input_format)
    if split == "test":
        test_label = list(map(int, pd_dataset["label"].values))
    else:
        test_label = pd_dataset["label"].values

    return test_dataset["id"], tokenized_test, test_label


def mark_entity_in_sentence(sent, input_format):
    ''' dataframe에서 하나의 row 내의 정보들을 조합해 마킹한 sentence를 만드는 함수'''
    # str 타입에서 dict 뽑아내기 
    sub = eval(sent['subject_entity'])
    obj = eval(sent['object_entity'])

    # 인덱스 뽑아와서 entity 구분하기
    indices = sorted([sub['start_idx'], sub['end_idx'], obj['start_idx'], obj['end_idx']])
    indices[1] += 1
    indices[3] += 1

    def split_string_by_index(string, indices):
        substrings = []
        start_index = 0
        for index in indices:
            substrings.append(string[start_index:index])
            start_index = index
        substrings.append(string[start_index:])
        return substrings

    split_sent = split_string_by_index(sent['sentence'], indices)

    # entity에 마킹하기
    lst = []
    if input_format == "entity_mask":
        for i in split_sent:
            if i == sub['word']:
                sub_token = f"[SUBJ-{sub['type']}]"
                lst.append(sub_token)
            elif i == obj['word']:
                obj_token = f"[OBJ-{obj['type']}]"
                lst.append(obj_token)
            else:
                lst.append(i)

    elif input_format == "entity_marker":
        for i in split_sent:
            if i == sub['word']:
                new_sub = ['[E1] '] + [sub['word']] + [' [/E1]']
                lst.append(new_sub)
            elif i == obj['word']:
                new_obj = ['[E2] '] + [obj['word']] + [' [/E2]']
                lst.append(new_obj)
            else:
                lst.append(i)

    elif input_format == "entity_marker_punct":
        for i in split_sent:
            if i == sub['word']:
                new_sub = ['@ '] + [sub['word']] + [' @']
                lst.append(new_sub)
            elif i == obj['word']:
                new_obj = ['# '] + [obj['word']] + [' #']
                lst.append(new_obj)
            else:
                lst.append(i)
    
    elif input_format == "typed_entity_marker":
        for i in split_sent:
            if i == sub['word']:
                new_sub = ['<S:'] + [sub['type']] + ['> '] + [sub['word']] + [' </S:'] + [sub['type']] + ['> ']
                lst.append(new_sub)
            elif i == obj['word']:
                new_obj = ['<O:'] + [obj['type']] + ['> '] + [obj['word']] + [' </O:'] + [obj['type']] + ['> ']
                lst.append(new_obj)
            else:
                lst.append(i)

    elif input_format == "typed_entity_marker_punct":
        for i in split_sent:
            if i == sub['word']:
                new_sub = ['@ '] + [' * '] + [sub['type'].lower()] + [' * '] + [sub['word']] + [' @ ']
                lst.append(new_sub)
            elif i == obj['word']:
                new_obj = [' # '] + [' ^ '] + [obj['type'].lower()] + [' ^ '] + [obj['word']] + [' # ']
                lst.append(new_obj)
            else:
                lst.append(i)
    # 최종 sentence로 만들고 공백 처리하기
    sentence = "".join(str(item) if isinstance(item, str) else "".join(item) for item in lst)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def preprocessing_dataset(dataset, input_format):
    """entity marking를 적용한 데이터를 저장합니다. """
    for i in tqdm(range(len(dataset))):
        dataset.loc[i, 'sentence'] = mark_entity_in_sentence(dataset.loc[i, :], input_format)

    dataset.drop(['subject_entity', 'object_entity', 'source'], axis=1, inplace=True)

    return dataset


def tokenized_dataset(dataset, tokenizer, input_format):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""

    # 새로운 특수 토큰 추가
    special_tokens = []
    
    if input_format == "entity_mask":
        special_tokens = ["[SUBJ-ORG]", "[SUBJ-PER]", "[OBJ-ORG]", "[OBJ-PER]", "[OBJ-LOC]", "[OBJ-DAT]", "[OBJ-POH]", "[OBJ-NOH]"]
    
    elif input_format == "entity_marker":
        special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    
    elif input_format == "typed_entity_marker":
        special_tokens = ["<S:PER>", "<S:ORG>", "<O:PER>", "<O:ORG>", "<O:LOC>", "<O:DAT>", "<O:POH>", "<O:NOH>",
                        "</S:PER>", "</S:ORG>", "</O:PER>", "</O:ORG>", "</O:LOC>", "</O:DAT>", "</O:POH>", "</O:NOH>"]
    
    tokenizer.add_tokens(special_tokens)

    tokenized_sentences = tokenizer(
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