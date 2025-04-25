import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import evaluate
from tqdm import tqdm
import torch
import json
import random
from datasets import Dataset
import pandas as pd

def load_data(data_path, instruction, train_num_samples=None):
    # Read the JSONL file 
    with open(data_path, 'r') as file:
        data_list = []
        for i, line in enumerate(file):
            data_list.append(json.loads(line))
    # Shuffle the data list
    random.shuffle(data_list)
    
    # If train_num_samples is provided, select only that many samples
    if train_num_samples is not None:
        data_list = data_list[:train_num_samples]

    # Convert list of dictionaries to dictionary of lists
    if data_list:
        data = {
            'instruction': [instruction for item in data_list],
            'input': [item.get('input', '') for item in data_list],
            'output': [item.get('output', '') for item in data_list]
        }
    else:
        # Fallback if file is empty
        test_data = {'instruction': [], 'input': [], 'output': []}
    print(f"Loaded {len(data['instruction'])} examples from {data_path}")
    
    return data


def create_eval_text(examples):
    """Assumes the input data is in JSON format with instruction, input, output text"""
    instruction = examples["instruction"]
    input_text = examples["input"]
    output_text = examples["output"]
    prompt = [f"<|begin_of_text|><|system|>{instruction}<|end_of_text|><|user|>{input_text}<|end_of_text|>" for instruction, input_text in zip(instruction, input_text)]
    label = [f"<|begin_of_text|><|assistant|>{output_text}<|end_of_text|>" for output_text in output_text]
    
    return {"text": prompt, "label": label}


def tokenize_and_truncate(examples, max_context_window, max_seq_len, mode="train"):
    tokenized = tokenizer(examples["text"], padding="max_length", max_length=max_seq_len)
    if mode == "eval":
        labels_tokenized = tokenizer(examples["label"])["input_ids"]
    # simple check for context window; just remove data points that are too long
    valid_indices = [i for i, ids in enumerate(tokenized["input_ids"]) if len(ids) <= max_context_window]
    tokenized["input_ids"] = [tokenized["input_ids"][i] for i in valid_indices]
    tokenized["attention_mask"] = [tokenized["attention_mask"][i] for i in valid_indices]
    if mode == "train":
        tokenized["labels"] = tokenized["input_ids"].copy()
    elif mode == "eval":
        tokenized["labels"] = labels_tokenized
    
    return tokenized


if __name__ == "__main__":
    batch_size = 64 # for production use, this should be set to 1 since there will be 1 user query at a time
    max_context_window = 8000
    max_seq_len = 350

    model_path = "./hf_cache/models/lora-0-100-train/checkpoint-390"
    test_data_path = "./hf_cache/datasets/test/test.jsonl"
    instruction_path = "./hf_cache/datasets/instruction.txt"
    tokenizer_path = "./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct-tokenizer"
    results_path = "./hf_cache/results"

    model_id = model_path.split("/")[-2]

    m = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(instruction_path, "r") as f:
        instruction = f.read()

    test_data = load_data(test_data_path, instruction)
    eval_ds = Dataset.from_dict(test_data)

    eval_text = eval_ds.map(
        create_eval_text,
        batched=True, num_proc=1,
        remove_columns=eval_ds.column_names,
    )
    eval_tokenized = eval_text.map(
    tokenize_and_truncate,
    batched=True, num_proc=1,
    remove_columns=eval_text.column_names,
    fn_kwargs={"max_context_window": max_context_window, "max_seq_len": max_seq_len, "mode": "eval"}
    )

    # Load the evaluation metric
    metric = evaluate.load("rouge")
    results = []

    for i in tqdm(range(0, len(eval_tokenized['input_ids']), batch_size)):
        batch_ids = torch.tensor(eval_tokenized['input_ids'][i:i+batch_size]).to(m.device)
        batch_masks = torch.tensor(eval_tokenized['attention_mask'][i:i+batch_size]).to(m.device)
        model_inputs = {"input_ids": batch_ids, "attention_mask": batch_masks}
        generated_ids = m.generate(**model_inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated_texts = tokenizer.batch_decode(generated_ids[:, batch_ids.shape[1]:], skip_special_tokens=True)
        
        # Process the results for this batch
        for gen_text, label in zip(generated_texts, eval_text["label"][i:i+batch_size]):
            result = metric.compute(predictions=[gen_text], references=[label], tokenizer=lambda x:x.split())
            results.append(result)

    # convert results to dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{results_path}/rouge-eval-{model_id}.csv", index=False)