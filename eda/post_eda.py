import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset
from sklearn.metrics import confusion_matrix


def make_dataframe(PATH: str, split: str, revision: int) -> pd.DataFrame:
    """
    주어진 경로(PATH)로부터의 데이터프레임과 Huggingface Datasets로부터 불러온 데이터프레임을 합친
    하나의 데이터프레임을 반환합니다.

    arguments:
        PATH (str): 'label'과 'pred_label' 열이 포함된 데이터프레임의 경로
        split (str): 데이터셋의 분할 유형 (train, validation, test).
        revision (str): 데이터셋의 버전 (commit hash).
    
    return:
        df (pd.DataFrame): 주어진 경로(PATH)로부터의 데이터프레임과 Huggingface Datasets으로부터 불러온 데이터프레임을 합친 
        하나의 데이터프레임
    """
    # 원본 validation set 불러오기
    valid = load_dataset("Smoked-Salmon-s/RE_Competition",
                        split = split,
                        column_names = ['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'],
                        revision = revision) 
    valid_df = valid.to_pandas().iloc[1:].reset_index(drop=True).astype({'id': 'int64'})

    # inference한 validation set 불러오기
    valid_inferred_df = pd.read_csv(PATH)
    
    # 두 dataframe 합치기
    df = pd.merge(valid_df, 
                valid_inferred_df[['id', 'pred_label', 'probs']],
                on='id',
                how='inner')
    df = df[['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'pred_label', 'probs', 'source']]

    return df


def confusion_matrix_graph(df: pd.DataFrame):
    """
    주어진 데이터프레임(df)의 'label'과 'pred_label' 열을 사용하여 confusion matrix을 계산하고
    heatmap 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'label'과 'pred_label' 열이 포함된 데이터프레임
    
    return:
        None. 함수는 confusion matrix heatmap을 출력합니다.
    """
    # confusion matrix 계산
    cm = confusion_matrix(df['label'], df['pred_label'])

    # 커스텀 컬러맵 생성
    cmap = mcolors.ListedColormap(['white', 'pink', 'tomato'])  

    # 정규화를 위한 경계값 설정
    bounds = [0.5, 1.0, 10.0, cm.max() + 0.5]  # 1.0, 10.0을 경계값으로 설정

    # 컬러맵을 적용할 값의 범위 설정
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 라벨 설정
    labels = sorted(list(df['label'].unique()))

    # heatmap 그리기
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, annot_kws={"size":8}, cmap=cmap, norm=norm, fmt='g', xticklabels=labels, yticklabels=labels)

    # 축 이름 및 제목 설정
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # 그래프 표시
    plt.show()


def all_label_matrix(df: pd.DataFrame, sort_column: str = 'label') -> pd.DataFrame:
    """
    주어진 데이터프레임(df)의 'label'과 'pred_label' 열을 사용하여 
    각 label에 대한 confusion matrix을 계산하고 dataframe 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'label'과 'pred_label' 열이 포함된 데이터프레임
        sort_column (str): confusion matrix dataframe을 정렬하는 기준 열
    
    return:
        metric_df (pd.DataFrame): confusion matrix dataframe
    """
    label_list = list(sorted(df['label'].unique()))

    label = [len(df[df['label'] == label]) for label in label_list]
    pred_label = [len(df[df['pred_label'] == label]) for label in label_list]
    TP = [len(df[(df['pred_label'] == label) & (df['label'] == label)]) for label in label_list]
    FP = [len(df[(df['pred_label'] == label) & (df['label'] != label)]) for label in label_list]
    FN = [len(df[(df['pred_label'] != label) & (df['label'] == label)]) for label in label_list]

    precision = []
    for tp, fp in zip(TP, FP):
        if tp + fp > 0:
            p = round(tp / (tp + fp), 4) 
        else:
            p = 0
        precision.append(p)

    recall = []
    for tp, fn in zip(TP, FN):
        if tp + fn > 0:
            r = round(tp / (tp + fn), 4)
        else:
            r = 0
        recall.append(r)

    metric_df = pd.DataFrame(zip(label_list, label, pred_label, TP, FP, FN, precision, recall))
    metric_df.columns = ['label', 'label_#', 'pred_label_#', 'TP', 'FP', 'FN', 'precision', 'recall']
    metric_df = metric_df.sort_values(sort_column)

    return metric_df


def specific_label_matrix(df: pd.DataFrame, label: str ='no_relation') -> pd.DataFrame:
    """
    주어진 데이터프레임(df)의 'label'과 'pred_label' 열을 사용하여 
    주어진 label에 대한 confusion matrix을 계산하고 dataframe 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'label'과 'pred_label' 열이 포함된 데이터프레임
        label (str): confusion matrix을 계산할 label
    
    return:
        metric_df (pd.DataFrame): 주어진 label에 대한 confusion matrix dataframe
    """
    TP = len(df[(df['pred_label'] == label) & (df['label'] == label)])
    FP = len(df[(df['pred_label'] == label) & (df['label'] != label)])
    FN = len(df[(df['pred_label'] != label) & (df['label'] == label)])

    precision = round(TP / (TP + FP), 4)
    recall = round(TP / (TP + FN), 4)

    metric_dict = {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall}
    metric_df = pd.DataFrame.from_dict(data = metric_dict, 
                                    orient='index', 
                                    columns=['value'])

    return metric_df


def total_metric(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 데이터프레임(df)의 'label'과 'pred_label' 열을 사용하여 
    전체 데이터에 대한 confusion matrix을 계산하고 dataframe 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'label'과 'pred_label' 열이 포함된 데이터프레임
    
    return:
        metric_df (pd.DataFrame): 주어진 데이터에 대한 confusion matrix dataframe
    """
    df = all_label_matrix(df)
    cleared_df = df[df['label'] != 'no_relation'].copy()
    
    TP = sum(cleared_df['TP'])
    FP = sum(cleared_df['FP'])
    FN = sum(cleared_df['FN'])
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    F1 = 2 * precision * recall / (precision + recall)

    metric_dict = {"TP": TP, "FP": FP, "FN": FN, "micro precision": precision, "micro recall": recall, " mircro F1 score": F1}
    metric_df = pd.DataFrame.from_dict(data = metric_dict, 
                                        orient='index', 
                                        columns=['value'])

    return metric_df


def precision_recall_graph(df: pd.DataFrame):
    """
    주어진 데이터프레임(df)의 'label'과 'pred_label' 열을 사용하여 
    각 label에 대한 precision과 recall을 계산하고 scatterplot 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'label'과 'pred_label' 열이 포함된 데이터프레임
    
    return:
        None. 함수는 precision과 recall에 대한 scatterplot을 출력합니다.
    """
    label_list = list(sorted(df['label'].unique()))

    label = [len(df[df['label'] == label]) for label in label_list]
    pred_label = [len(df[df['pred_label'] == label]) for label in label_list]
    TP = [len(df[(df['pred_label'] == label) & (df['label'] == label)]) for label in label_list]
    FP = [len(df[(df['pred_label'] == label) & (df['label'] != label)]) for label in label_list]
    FN = [len(df[(df['pred_label'] != label) & (df['label'] == label)]) for label in label_list]

    precision = []
    for tp, fp in zip(TP, FP):
        if tp + fp > 0:
            p = round(tp / (tp + fp), 4) 
        else:
            p = 0
        precision.append(p)

    recall = []
    for tp, fn in zip(TP, FN):
        if tp + fn > 0:
            r = round(tp / (tp + fn), 4)
        else:
            r = 0
        recall.append(r)

    plt.scatter(recall, precision)

    # 그래프 제목과 축 레이블 설정
    plt.title('relation between recall and precision')
    plt.xlabel('recall')
    plt.ylabel('precision')

    # 그래프 보이기
    plt.show()