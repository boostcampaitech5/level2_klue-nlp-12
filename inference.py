import sys
import torch
import numpy as np
import pandas as pd
import pickle as pickle
import torch.nn.functional as F
from tqdm import tqdm
from load_data import *
from utils import *
from args import parse_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def inference(model, tokenized_sent, device):
    seed_everything(config.seed)
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()

    output_pred = []
    output_prob = []

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                token_type_ids=data["token_type_ids"].to(device),
            )

        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []

    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)

    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def main(config):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    Tokenizer_NAME = config.arch["type"]
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # load my model
    MODEL_NAME = config.trainer["model_dir"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    # load test dataset
    revision = config.dataloader["revision"]
    test_id, test_dataset, test_label = load_test_dataset("test", revision, tokenizer)
    re_test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    pred_answer, output_prob = inference(
        model, re_test_dataset, device
    )  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환

    # make csv file with predicted answer
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    output_path = config.trainer["pred_dir"]  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장

    ## 사후분석을 위한 validation data inference
    # load validation dataset
    val_id, val_dataset, val_label = load_test_dataset("validation", revision, tokenizer)
    Re_val_dataset = RE_Dataset(val_dataset, [100] * len(val_id))

    # predict validation answer
    pred_val_answer, val_output_prob = inference(
        model, Re_val_dataset, device
    )
    pred_val_answer = num_to_label(pred_val_answer)

    # make csv file with predicted validation answer
    val_output = pd.DataFrame(
        {
            "id": val_id,
            "true_label": val_label,
            "pred_label": pred_val_answer,
            "probs": val_output_prob,
        }
    )
    val_output_path = config.trainer["val_pred_dir"]  
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
    val_output.to_csv(val_output_path, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장

    print("---- Finish! ----")


if __name__ == "__main__":

    try:
        config_path = sys.argv[1]
    except:
        config_path = './config.yaml'
        
    config = parse_arguments(config_path)

    main(config)
