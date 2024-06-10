from transformers import AutoConfig, AutoModel, PreTrainedModel
import torch.nn as nn
import torch


class CustomModel(PreTrainedModel):
    def __init__(self, model_name, num_extra_dims, num_labels):
        config = AutoConfig.from_pretrained(model_name)
        super(CustomModel, self).__init__(config)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(num_hidden_size + num_extra_dims, num_labels)

    def forward(self, input_ids, attention_mask=None, extra_data=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        concat = torch.cat((cls_embeds, extra_data), dim=-1)
        logits = self.classifier(concat)

        return logits
