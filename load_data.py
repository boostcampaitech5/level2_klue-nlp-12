import pickle as pickle
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from utils import *


def load_train_dataset(split, revision, tokenizer, input_format=None, prompt=None, type_transform=False):
    """train dataset을 불러온 후, tokenizing 합니다."""

    if input_format is None:
        input_format = 'default'
    if prompt is None:
        prompt = 'default'
    print('input format: ',input_format, 'prompt: ', prompt)

    dataset = load_dataset(
        'Smoked-Salmon-s/RE_Competition',
        split=split,
        column_names=['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'],
        revision=revision,
    )
    pd_dataset = dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({'id': 'int64'})
    train_dataset = preprocessing_dataset(pd_dataset, input_format, type_transform)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, input_format, prompt)
    train_label = pd_dataset['label'].values

    return tokenized_train, train_label


def load_test_dataset(split, revision, tokenizer, input_format=None, prompt=None, type_transform=False):
    """test dataset을 불러온 후, tokenizing 합니다."""

    if input_format is None:
        input_format = 'default'
    if prompt is None:
        prompt = 'default'
    print('input format: ',input_format, 'prompt: ', prompt)

    dataset = load_dataset(
        'Smoked-Salmon-s/RE_Competition',
        split=split,
        column_names=['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'],
        revision=revision,
    )
    pd_dataset = dataset.to_pandas().iloc[1:].reset_index(drop=True).astype({'id': 'int64'})
    test_dataset = preprocessing_dataset(pd_dataset, input_format, type_transform)
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, input_format, prompt)
    
    if split == 'test':
        test_label = list(map(int, pd_dataset['label'].values))
    else:
        test_label = pd_dataset['label'].values

    return test_dataset['id'], tokenized_test, test_label


def preprocessing_dataset(dataset, input_format, type_transform=False):
    """subject_entity column과 object_entity column을 리스트 형태로 변환하고, 
        sentence column에 marker를 적용합니다."""
    
    subject_entity = []
    object_entity = []

    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        i = i[1:-1].split(',')[0].split(':')[1]
        j = j[1:-1].split(',')[0].split(':')[1]
        subject_entity.append(i)
        object_entity.append(j)

    dataset['subj_entity'] = subject_entity
    dataset['obj_entity'] = object_entity

    # entity type을 한글로 번역
    if type_transform:
        print('entity type을 한글로 번역합니다.')
        hanguled = [to_hangul(row_data) for index, row_data in tqdm(dataset.iterrows())]
        dataset['subject_entity'] = [x[0] for x in hanguled]
        dataset['object_entity'] = [x[1] for x in hanguled]

    input_format_list = ['entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct']
    if input_format in input_format_list:
        marked_sentences = [marker(row_data, input_format) for index, row_data in tqdm(dataset.iterrows())]
        dataset['sentence'] = marked_sentences
    elif input_format == 'default':
        pass
    else:
        raise ValueError('잘못된 input_format이 입력되었습니다. ')

    return dataset


def tokenized_dataset(dataset, tokenizer, input_format, prompt):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    # 새로운 특수 토큰 추가
    special_tokens = []
    
    if input_format == 'entity_mask':
        special_tokens = ['[SUBJ-ORG]', '[SUBJ-PER]', '[OBJ-ORG]', '[OBJ-PER]', '[OBJ-LOC]', '[OBJ-DAT]', '[OBJ-POH]', '[OBJ-NOH]']
    
    elif input_format == 'entity_marker':
        special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
    
    elif input_format == 'typed_entity_marker':
        special_tokens = ['<S:PER>', '<S:ORG>', '<O:PER>', '<O:ORG>', '<O:LOC>', '<O:DAT>', '<O:POH>', '<O:NOH>',
                        '</S:PER>', '</S:ORG>', '</O:PER>', '</O:ORG>', '</O:LOC>', '</O:DAT>', '</O:POH>', '</O:NOH>']
    
    tokenizer.add_tokens(special_tokens)

    # prompt 추가
    if prompt in ['s_sep_o', 's_and_o']:
        concat_entity = []

        if prompt == 's_sep_o':
            for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
                temp = ''
                temp = e01[2:-1] + '[SEP]' + e02[2:-1]
                concat_entity.append(temp)

        elif prompt == 's_and_o':
            for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
                temp = ''
                temp = e01[2:-1] + '와 ' + e02[2:-1] + '의 관계'
                concat_entity.append(temp)

        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
            
    elif prompt == 'default':
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )

    else:
        raise ValueError('잘못된 prompt가 입력되었습니다. ')

    return tokenized_sentences


def label_to_num(label):
    """원본 문자열 label을 숫자 형식 class로 변환."""

    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label):
    """숫자 형식 class를 원본 문자열 label로 변환."""

    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


class REDataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
