import pickle as pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Tuple, List

from utils.args import *
from load_data.load_data import *
from model.model import *
from utils.utils import *


def inference(model: torch.nn.Module, tokenized_sent: DataLoader, device: torch.device) -> Tuple[List[int], List[List[float]]]:
    """
    test dataset을 DataLoader로 만들어 준 후 batch_size로 나눠 model이 예측

    Args:
        model (torch.nn.Module): 예측에 사용할 모델
        tokenized_sent (DataLoader): 토큰화가 완료된 문장 데이터셋
        device (torch.device): 모델을 실행할 디바이스 (예: cuda:0)

    Returns:
        Tuple[List[int], List[List[float]]]: 예측된 클래스 인덱스와 각 클래스에 대한 확률이 담긴 리스트를 반환
    """

    dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
    model.eval()

    output_pred = []
    output_prob = []

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device),
            )

        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def main() -> None:
    """
    주어진 데이터셋 csv 파일과 같은 형태일 경우 inference를 수행할 수 있는 메인 함수

    다음 프로세스를 수행:
        1. config에 따라 시드를 고정하고, 디바이스를 설정
        2. 토크나이저와 모델을 로드하고, 학습시킨 모델을 로드
        3. 테스트 데이터셋을 로드하고, 데이터셋 객체 생성
        4. 모델을 이용하여 예측 수행
        5. 예측 결과를 csv 파일로 저장
        6. full train이 아닐 경우 검증 데이터셋에 대해서도 같은 과정을 수행

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

    seed_everything(config.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load tokenizer
    model_name = config.model['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load my model
    model_module = __import__('model.model', fromlist=[config.model['variant']])
    model_class = getattr(model_module, config.model['variant'])
    # Available customized classes:
    #   BaseREModel, BiLSTMREModel, BiGRUREModel
    model = model_class(config, len(tokenizer))

    load_model_path = './best_model/pytorch_model.bin'
    checkpoint = torch.load(load_model_path)
    model.load_state_dict(checkpoint)

    model.parameters
    model.to(device)

    # load test dataset
    revision = config.dataloader['revision']
    input_format = config.dataloader.get('input_format')
    prompt = config.dataloader.get('prompt')
    type_transform = config.dataloader.get('type_transform')

    test_id, test_dataset, test_label = load_test_dataset(
        split='test',
        revision=revision,
        tokenizer=tokenizer,
        input_format=input_format,
        prompt=prompt,
        type_transform=type_transform,
    )
    re_test_dataset = REDataset(test_dataset, test_label)

    # predict answer
    pred_answer, output_prob = inference(model, re_test_dataset, device)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            'id': test_id,
            'pred_label': pred_answer,
            'probs': output_prob,
        }
    )
    output_path = config.trainer['pred_dir']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장

    ## 사후분석을 위한 validation data inference
    # load validation dataset(full train일 경우 revision에 valid가 없어서 load_test_dataset에서 오류가 생기므로 넘기기)
    try:
        val_id, val_dataset, val_label = load_test_dataset(
            split=config.dataloader['valid_split'],
            revision=revision,
            tokenizer=tokenizer,
            input_format=input_format,
            prompt=prompt,
            type_transform=type_transform,
        )
        re_val_dataset = REDataset(val_dataset, [100] * len(val_id))

        # predict validation answer
        pred_val_answer, val_output_prob = inference(model, re_val_dataset, device)
        pred_val_answer = num_to_label(pred_val_answer)

        # make csv file with predicted validation answer
        val_output = pd.DataFrame(
            {
                'id': val_id,
                'true_label': val_label,
                'pred_label': pred_val_answer,
                'probs': val_output_prob,
            }
        )
        val_output_path = config.trainer['val_pred_dir']
        os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
        val_output.to_csv(val_output_path, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장
        
    except ValueError:
        print('There is no existing valiation dataset. The inference output is from full dataset model.')

    print('---- Finish! ----')


if __name__ == '__main__':
    main()