from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')
model = AutoModelForSequenceClassification.from_pretrained('mrm8488/codebert-base-finetuned-detect-insecure-code')

myCode = """
pragma solidity ^0.8.0;

contract IntegerOverflowUnderflowExample {
    uint256 public maxValue = 2**256 - 1;

    function triggerOverflow() public view returns (uint256) {
        uint256 result = maxValue + 1;
        return result;
    }

    function triggerUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}



"""
inputs = tokenizer(myCode, return_tensors="pt", truncation=True, padding='max_length')
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(np.argmax(logits.detach().numpy()))
