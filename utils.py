import logging
import os
import re
import random

import numpy as np
import torch
import wandb
from wandb import AlertLevel

log = logging.getLogger(__name__)


def seed_everything(seed, workers: bool = False) -> int:
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    return seed


def init_wandb(config, run_name):
    if not config.use_wandb:
        return

    wandb.init(
        entity=config.wandb['entity'],
        project=config.wandb['project_name'],
        name=run_name,
        config=config,
    )
    wandb.alert(title='start', level=AlertLevel.INFO, text=f'{run_name}')


def alert_wandb(config, run_name, title):
    if config.use_wandb:
        wandb.alert(title=title, level=AlertLevel.INFO, text=f'{run_name}')


def to_hangul(sent):
        dic = {
            "ORG" : "조직",
            "PER" : "사람",
            "DAT" : "시간",
            "LOC" : "장소",
            "POH" : "기타",
            "NOH" : "수량",
        }
        
        sub = eval(sent['subject_entity'])
        obj = eval(sent['object_entity'])

        sub['type'] = dic[sub['type']]
        obj['type'] = dic[obj['type']]

        sent['subject_entity'] = str(sub)
        sent['object_entity'] = str(obj)
        
        return sent['subject_entity'], sent['object_entity']


def marker(sent, input_format):
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
