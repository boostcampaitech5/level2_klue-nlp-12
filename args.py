import yaml
from argparse import Namespace


def parse_arguments() -> Namespace:
    """config.json 파일의 내용을 argparse.Namespace 객체로 변환.

    Returns:
        args (argparse.Namespace): config.json 파일의 내용을 포함하는 Namespace 객체.
    """

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = Namespace(**config)
    return args