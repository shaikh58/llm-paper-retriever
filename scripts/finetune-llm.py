import importlib
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline, LogitsProcessorList, EarlyStoppingCallback
from datasets import load_dataset, Dataset, IterableDataset
import evaluate
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import outlines
import random
import json
from pydantic import BaseModel
from typing import Literal
random.seed(42)

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

def create_instruct_text(examples):
    """Assumes the input data is in JSON format with instruction, input, output text"""
    instruction = examples["instruction"]
    input_text = examples["input"]
    output_text = examples["output"]
    formatted = [f"<|begin_of_text|><|system|>{instruction}<|end_of_text|><|user|>{input_text}<|end_of_text|><|assistant|>{output_text}<|end_of_text|>" for instruction, input_text, output_text in zip(instruction, input_text, output_text)]
    
    return {"text": formatted}

def create_eval_text(examples):
    """Assumes the input data is in JSON format with instruction, input, output text"""
    instruction = examples["instruction"]
    input_text = examples["input"]
    output_text = examples["output"]
    prompt = [f"<|begin_of_text|><|system|>{instruction}<|end_of_text|><|user|>{input_text}<|end_of_text|>" for instruction, input_text in zip(instruction, input_text)]
    label = [f"<|begin_of_text|><|assistant|>{output_text}<|end_of_text|>" for output_text in output_text]
    
    return {"text": prompt, "label": label}


def tokenize_and_truncate(examples, max_context_window, max_seq_len, mode="train"):
    if mode == "train":
        tokenized = tokenizer(examples["text"], padding="max_length", max_length=max_seq_len)
    elif mode == "eval":
        tokenized = tokenizer(examples["text"])
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
    TRAIN_NUM_SAMPLES = 50000
    model_save_dir = "./hf_cache/models"
    test_data_path = "./hf_cache/datasets/test/test.jsonl"
    train_data_path = "./hf_cache/datasets/train/train.jsonl"
    instruction_path = "./hf_cache/datasets/instruction.txt"
    tokenizer_path = "./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct-tokenizer"
    os.environ["WANDB_PROJECT"] = "llm"
    run_name = "lora-0-50k-train-grad-accum"
    max_seq_len = 350
    max_context_window = 8000 # llama3-8b-instruct official context window

    model = AutoModelForCausalLM.from_pretrained(
        # "meta-llama/Meta-Llama-3-8B-Instruct", # only when downloading for the first time
        "./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    # run this the first time you download the model; then, load in from this directory
    # model.save_pretrained("./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct")

    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(instruction_path, "r") as f:
        instruction = f.read()
    train_data = load_data(train_data_path, instruction, TRAIN_NUM_SAMPLES)
    test_data = load_data(test_data_path, instruction)

    ds = Dataset.from_dict(train_data)
    eval_ds = Dataset.from_dict(test_data)

    instruct_text = ds.map(
        create_instruct_text,
        batched=True,num_proc=1,
        remove_columns=ds.column_names,
    )
    tokenized = instruct_text.map(
        tokenize_and_truncate, 
        batched=True, num_proc=1,
        remove_columns=instruct_text.column_names,
        fn_kwargs={"max_context_window": max_context_window, "max_seq_len": max_seq_len})
    eval_text = eval_ds.map(
        create_eval_text,
        batched=True, num_proc=1,
        remove_columns=eval_ds.column_names,
    )
    eval_tokenized = eval_text.map(
        tokenize_and_truncate,
        batched=True, num_proc=1,
        remove_columns=eval_text.column_names,
        fn_kwargs={"max_context_window": max_context_window, "max_seq_len": max_seq_len}
    )
    max_len = 0
    for i, ids in enumerate(tokenized["input_ids"]):
        max_len = max(max_len, len(ids))
    print("Max length of tokenized dataset: ", max_len)

    # this collator dynamically pads the sequences to the max length of the batch; no need to define our own attention mask
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # type of task to train on
        inference_mode=False, # set to False for training
        r=16, # dimension of the smaller matrices A,B; e.g. hidden_size = 4096 -> (4096, 8)
        lora_alpha=32, # scaling factor
        lora_dropout=0.05 # dropout of LoRA layers
    )
    # model.delete_adapter("lora_1")
    model.add_adapter(lora_config)

    training_args = TrainingArguments(
        output_dir=os.path.join(model_save_dir, run_name),
        num_train_epochs=30,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        # logging_dir="./hf_cache/logs",
        logging_steps=5,
        run_name=run_name,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        torch_empty_cache_steps=100,
        report_to="wandb",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)]
    )

    trainer.train()