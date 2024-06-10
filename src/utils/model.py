from transformers import AutoConfig, AutoModel, PreTrainedModel
from typing import Dict
import torch.nn as nn
import torch


class CustomModel(PreTrainedModel):
    def __init__(self, config: Dict):
        self.training_config = config

        config = AutoConfig.from_pretrained(self.training_config["model_name"])
        super(CustomModel, self).__init__(config)
        self.num_labels = self.training_config["num_labels"]
        self.transformer = AutoModel.from_pretrained(self.training_config["model_name"], config=config)
        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(num_hidden_size + 40, self.num_labels)

    def forward(self, input_ids, attention_mask=None, extra_data=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        concat = torch.cat((cls_embeds, extra_data), dim=-1)
        logits = self.classifier(concat)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
