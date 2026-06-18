import torch


class TextGenerator:
    def __init__(self, loader):
        self.loader = loader
        self.model = loader.model
        self.tokenizer = loader.tokenizer

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64):
        ids = self.loader.encode(prompt)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def perplexity(self, text):
        ids = self.loader.encode(text)
        out = self.model(ids, labels=ids)
        return float(torch.exp(out.loss))
