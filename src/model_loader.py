import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class MambaLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id, dtype=_DTYPES[config.dtype]
        ).to(config.device)
        self.model.lm_head.weight = self.model.get_input_embeddings().weight
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def selective_linears(self):
        out = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "mixer" in name:
                out[name] = module
        return out

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.config.device)