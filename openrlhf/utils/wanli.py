import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class RobertaLargeWanli:
    def __init__(self, device="cuda"):
        self.model = RobertaForSequenceClassification.from_pretrained('alisawuffles/roberta-large-wanli', device_map=device)
        self.tokenizer = RobertaTokenizer.from_pretrained('alisawuffles/roberta-large-wanli')

    def compute_wanli_score(self, original, revised, mode="non_contradiction"):
        """
        Compute WANLI score between original and revised prompts.
        
        return a score that presents the label probability of non-contradiction

        model.config.id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'} 
        """
        x = self.tokenizer(original, revised, return_tensors='pt', max_length=128, truncation=True).to(self.model.device)
        logits = self.model(**x).logits
        probs = logits.softmax(dim=1).squeeze(0)
        if mode == "non_contradiction":
            output = 1. - probs[0].item()
        elif mode == "contradiction":
            output = probs[0].item()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return output
