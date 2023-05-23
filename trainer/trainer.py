import torch
from transformers import Trainer


class RETrainer(Trainer):
    def __init__(self, *args, loss_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg

    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        # 인덱스에 맞춰서 과거 ouput을 다 저장
        if self.args.past_index >= 0:
            self._past= outputs[self.args.past_index]

        # 커스텀 loss 정의
        if self.loss_cfg['type'] == 'CrossEntropyLoss':
            loss_fct = torch.nn.functional.cross_entropy
        elif self.loss_cfg['type'] == 'WeightedCrossEntropyLoss':
            loss_fct = torch.nn.CrossEntropyLoss(weight = torch.Tensor(self.loss_cfg['weights']).to(device))
        else:
            loss_module = __import__('model.loss', fromlist=[self.loss_cfg['type']])
            loss_class = getattr(loss_module, self.loss_cfg['type'])
            if self.loss_cfg['type'] == 'LovaszSoftmaxLoss':
                loss_fct = loss_class()
            elif self.loss_cfg['type'] == 'FocalLoss':
                loss_fct = loss_class(self.loss_cfg['focal_alpha'], self.loss_cfg['focal_gamma'])
            elif self.loss_cfg['type'] == 'WeightedFocalLoss':
                loss_fct = loss_class(alpha = torch.Tensor(self.loss_cfg['weight_focal_alpha']).to(device), gamma= self.loss_cfg['focal_gamma'])
            elif self.loss_cfg['type'] == 'MulticlassDiceLoss':
                loss_fct = loss_class(self.loss_cfg['dice_smooth'])
            else:
                raise ValueError('Unsupported loss type')

        # Check the type of outputs and extract logits
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]

        loss = loss_fct(logits, labels).to(device)
        return (loss, outputs) if return_outputs else loss
