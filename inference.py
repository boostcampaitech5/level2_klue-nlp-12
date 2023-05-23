import pickle as pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.args import *
from load_data.load_data import *
from model.model import *
from utils.utils import *


def inference(model, tokenized_sent, device: torch.device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
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


def main():
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
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