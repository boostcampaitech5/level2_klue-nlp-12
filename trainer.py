import torch
from transformers import Trainer


class RETrainer(Trainer):
    def __init__(self, loss_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg

    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')

        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]

        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >= 0:
            self._past= outputs[self.args.past_index]

        # 커스텀 loss 정의
        if self.loss_cfg['type'] == 'CrossEntropyLoss':
            loss_fct = torch.nn.functional.cross_entropy
        else:
            loss_module = __import__('loss', fromlist=[self.loss_cfg['type']])
            loss_class = getattr(loss_module, self.loss_cfg['type'])
            if self.loss_cfg['type'] == 'LovaszSoftmaxLoss':
                loss_fct = loss_class()
            elif self.loss_cfg['type'] == 'FocalLoss':
                loss_fct = loss_class(self.loss_cfg['focal_alpha'], self.loss_cfg['focal_gamma'])
            elif self.loss_cfg['type'] == 'MulticlassDiceLoss':
                loss_fct = loss_class(self.loss_cfg['dice_smooth'])
            else:
                raise ValueError('Unsupported loss type')
        loss = loss_fct(logits, labels).to(device)

        return (loss, outputs) if return_outputs else loss
