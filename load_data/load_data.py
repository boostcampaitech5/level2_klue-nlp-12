import pickle as pickle
import re

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

from utils.utils import *


def load_train_dataset(
    split: str,
    revision: str,
    tokenizer: PreTrainedTokenizer,
    input_format: str = None,
    prompt: str = None,
    type_transform: bool = False
) -> Tuple[Dict[str, Union[List[str], List[int]]], Union[int, List[str]]]:
    """
    train dataset을 불러온 후, tokenizing 하는 함수입니다.

    Args:
        split (str): 데이터셋의 분할 유형 (train, validation, test).
        revision (str): 데이터셋의 버전 (commit hash).
        tokenizer (PreTrainedTokenizer): 사용할 토크나이저 객체.
        input_format (str, optional): entity representation 유형. 기본값은 None이며, default로 설정됩니다.
        prompt (str, optional): prompt 유형. 기본값은 None이며, default로 설정됩니다.
        type_transform (bool, optional): entity type을 한글로 번역할지 여부. 기본값은 False입니다.

    Returns:
        Tuple[Dict[str, Union[List[str], List[int]]], Union[int, List[str]]]
        : 토큰화된 train 데이터셋과 레이블.
    """
    
    if input_format is None:
        input_format = 'default'
    if prompt is None:
        prompt = 'default'
    print('input format: ',input_format, '| prompt: ', prompt)

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


def load_test_dataset(
    split: str,
    revision: str,
    tokenizer: PreTrainedTokenizer,
    input_format: str = None,
    prompt: str = None,
    type_transform: bool = False
) -> Tuple[Union[int, str], Dict[str, Union[List[str], List[int]]], Union[int, List[str]]]:
    """
    test dataset을 불러온 후, tokenizing 하는 함수입니다.

    Args:
        split (str): 데이터셋의 분할 유형 (train, validation, test).
        revision (str): 데이터셋의 버전 (commit hash).
        tokenizer (PreTrainedTokenizer): 사용할 토크나이저 객체.
        input_format (str, optional): entity representation 유형. 기본값은 None이며, default로 설정됩니다.
        prompt (str, optional): prompt 유형. 기본값은 None이며, default로 설정됩니다.
        type_transform (bool, optional): entity type을 한글로 번역할지 여부. 기본값은 False입니다.

    Returns:
        Tuple[Union[int, str], Dict[str, Union[List[str], List[int]]], Union[int, List[str]]]
        : test 데이터셋의 id, 토큰화된 문장, 레이블.
    """

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


def preprocessing_dataset(
    dataset: Dict[str, List[str]],
    input_format: str,
    type_transform: bool = False
) -> Dict[str, List[str]]:
    """
    subject_entity column과 object_entity column을 리스트 형태로 변환하고, 
    sentence column에 entity representation를 적용하는 함수입니다.

    Args:
        dataset (Dict[str, List[str]]): 전처리할 데이터셋.
        input_format (str): entity representation 유형.
        type_transform (bool, optional): entity type을 한글로 번역할지 여부. 기본값은 False입니다.

    Returns:
        Dict[str, List[str]]: 전처리된 데이터셋.
    """

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

    # entity representation 적용
    input_format_list = ['entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct']
    if input_format in input_format_list:
        marked_sentences = [marker(row_data, input_format) for index, row_data in tqdm(dataset.iterrows())]
        dataset['sentence'] = marked_sentences
    elif input_format == 'default':
        pass
    else:
        raise ValueError('잘못된 input_format이 입력되었습니다. ')

    return dataset


def tokenized_dataset(
    dataset: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    input_format: str,
    prompt: str
) -> Dict[str, Union[List[str], List[int]]]:
    """
    tokenizer에 따라 문장을 토큰화하는 함수입니다.

    Args:
        dataset (Dict[str, List[str]]): 토큰화할 데이터셋.
        tokenizer (PreTrainedTokenizer): 사용할 토크나이저 객체.
        input_format (str): entity representation 유형.
        prompt (str): prompt 유형.

    Returns:
        Dict[str, Union[List[str], List[int]]]: 토큰화된 문장의 딕셔너리.
    """

    # 새로운 특수 토큰 추가
    special_tokens = []
    
    if input_format == 'entity_mask':
        special_tokens = ['[SUBJ-ORG]', '[SUBJ-PER]', '[OBJ-ORG]', '[OBJ-PER]', '[OBJ-LOC]', '[OBJ-DAT]', '[OBJ-POH]', '[OBJ-NOH]']

    elif input_format == 'entity_marker':
        special_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']

    elif input_format == 'typed_entity_marker':
        special_tokens = ['<S:PER>', '<S:ORG>', '<O:PER>', '<O:ORG>', '<O:LOC>', '<O:DAT>', '<O:POH>', '<O:NOH>',
                        '</S:PER>', '</S:ORG>', '</O:PER>', '</O:ORG>', '</O:LOC>', '</O:DAT>', '</O:POH>', '</O:NOH>']

    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # check
    print("length of tokenizer:", len(tokenizer))
    print("length of special tokens: ", tokenizer.all_special_tokens)
    print("special tokens:", tokenizer.special_tokens_map)

    # prompt 추가
    if prompt in ['s_sep_o', 's_and_o', 'quiz']:
        prompt_forward = []

        if prompt == 's_sep_o':
            for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
                temp = ''
                temp = e01[2:-1] + '[SEP]' + e02[2:-1]
                prompt_forward.append(temp)

        elif prompt == 's_and_o':
            for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
                temp = ''
                temp = e01[2:-1] + '와 ' + e02[2:-1] + '의 관계'
                prompt_forward.append(temp)
        
        elif prompt == 'quiz':
            for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
                temp = ''
                temp = '다음 문장에서 ' + e01[2:-1] + '와 ' + e02[2:-1] + '사이의 관계를 추출하시오.'
                prompt_forward.append(temp)       

        tokenized_sentences = tokenizer(
            prompt_forward,
            list(dataset['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=180,
            add_special_tokens=True,
        )
    
    elif prompt == 'problem':
        prompt_forward = []
        prompt_backward = []

        for e01, e02 in zip(dataset['subj_entity'], dataset['obj_entity']):
            temp = ''
            temp = '다음 문장에서 ' + e01[2:-1] + '와 ' + e02[2:-1] + '사이의 관계를 추출하시오.'
            prompt_forward.append(temp)
        for e00, e01, e02 in zip(dataset['sentence'], dataset['subj_entity'], dataset['obj_entity']):
            temp = ''
            temp = e00 + e01[2:-1] + '와 ' + e02[2:-1] + '는 어떤 관계입니까?'
            prompt_backward.append(temp)  

        tokenized_sentences = tokenizer(
            prompt_forward,
            prompt_backward,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200,
            add_special_tokens=True,
        )  
            
    elif prompt == 'default':
        tokenized_sentences = tokenizer(
            list(dataset['sentence']),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=180,
            add_special_tokens=True,
        )

    else:
        raise ValueError('잘못된 prompt가 입력되었습니다. ')

    return tokenized_sentences


def label_to_num(label: List[str]) -> List[int]:
    """
    원본 문자열 label을 숫자 형식 클래스로 변환하는 함수입니다.
    
    Args:
        label (List[str]): 변환할 원본 문자열 클래스 리스트.
        
    Returns:
        List[int]: 숫자 형식으로 변환된 클래스 리스트.
    """

    num_label = []
    with open('load_data/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def num_to_label(label: List[int]) -> List[str]:
    """
    숫자 형식 클래스를 원본 문자열 label로 변환하는 함수입니다.
    
    Args:
        label (List[int]): 변환할 숫자 형식의 클래스 리스트.
        
    Returns:
        List[str]: 원본 문자열로 변환된 클래스 리스트.
    """

    origin_label = []
    with open('load_data/dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


class REDataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class입니다."""

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