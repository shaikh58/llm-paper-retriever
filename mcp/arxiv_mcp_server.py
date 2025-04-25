from fastmcp import FastMCP
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
import re
from typing import Optional, Literal
from pydantic import BaseModel

# TODO: specify dependencies
mcp = FastMCP("arxiv-mcp-server", dependencies=["transformers", "datasets", "pydantic", "torch", "typing", "json"])

# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Meta-Llama-3-8B-Instruct",
#     # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#     device_map="auto",
#     local_files_only=True
#  )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define the valid operators as string literals
ComparisonOperatorType = Literal[">", "<", ">=", "<=", "="]

class CitationCriteria(BaseModel):
    operator: ComparisonOperatorType
    value: int
    
    @classmethod
    def parse_citation_string(cls, citation_str: str) -> "CitationCriteria":
        # Remove any whitespace
        citation_str = citation_str.replace(" ", "")
        
        # Match patterns like >=50, >100, =75, etc.
        pattern = r'([><=]=?|<)(\d+)'
        match = re.match(pattern, citation_str)
        
        if match:
            operator_str, value_str = match.groups()
            # The operator_str is already in the correct format
            return cls(operator=operator_str, value=int(value_str))
        raise ValueError(f"Invalid citation criteria format: {citation_str}")

class Query(BaseModel):
    topic: Optional[str] = None
    journal: Optional[str] = None
    author: Optional[str] = None
    citations: Optional[CitationCriteria] = None
    keyword: Optional[str] = None
    limit: Optional[int] = 10
    sort_by: Optional[Literal["date", "citations", "relevance"]] = "relevance"
    sort_order: Optional[Literal["ascending", "descending"]] = "descending"


def markdown_to_json(markdown_text: str) -> str:
    """Converts markdown from model output to json for arxiv query"""
    # Initialize an empty dictionary to store our parameters
    params = {}
    
    # Helper function to extract value from a line
    def extract_value(line: str) -> str:
        return line.split("**: ", 1)[1].strip()
    
    # Split the text into sections
    sections = markdown_text.split("##")
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split("\n")
        section_title = lines[0].strip().lower()
        
        # Process each bullet point in the section
        for line in lines[1:]:
            if not line.strip() or not "**" in line:
                continue
                
            # Extract the parameter name and value
            param_name = line.split("**")[1].lower()
            value = extract_value(line)
            
            if param_name == "topic":
                params["topic"] = value
            elif param_name == "author":
                params["author"] = value
            elif param_name == "journal":
                params["journal"] = value
            elif param_name == "citations":
                params["citations"] = CitationCriteria.parse_citation_string(value)
            elif param_name == "keyword":
                params["keyword"] = value
            elif param_name == "limit":
                params["limit"] = int(value)
            elif param_name == "sort by":
                params["sort_by"] = value.lower()
            elif param_name == "sort order":
                params["sort_order"] = value.lower()
    
    return Query(**params).model_dump_json()

def request_arxiv(query: str) -> str:
    """Requests arxiv papers"""
    return query

def parse_arxiv_response(response: str) -> str:
    """Parses arxiv response"""
    return response

def load_data(data_path: str, instruction_path: str) -> dict:

    with open(instruction_path, "r") as f:
        instruction = f.read()

    # Read the JSONL file 
    with open(data_path, 'r') as file:
        data_list = [json.loads(line) for line in file]

    # Convert list of dictionaries to dictionary of lists
    if data_list:
        data = {
            'instruction': [instruction for item in data_list],
            'input': [item.get('input', '') for item in data_list],
            'output': [item.get('output', '') for item in data_list]
        }
    else:
        # Fallback if file is empty
        data = {'instruction': [], 'input': [], 'output': []}
    
    return data

def prepare_query(query: str, instruction_path: str) -> dict:
    """Convert string query into a JSON format for input to preprocessing pipeline"""
    with open(instruction_path, "r") as f:
        instruction = f.read()
    data = {'instruction': [instruction], 'input': [query]}

    return data

def create_inference_text(examples):
    """Assumes the input data is in JSON format with instruction, input, output text"""
    instruction = examples["instruction"]
    input_text = examples["input"]
    prompt = [f"<|begin_of_text|><|system|>{instruction}<|end_of_text|><|user|>{input_text}<|end_of_text|>" for instruction, input_text in zip(instruction, input_text)]
    
    return {"text": prompt}


def tokenize_and_truncate(examples, max_context_window, max_seq_len):
    tokenized = tokenizer(examples["text"])
    # simple check for context window; just remove data points that are too long
    valid_indices = [i for i, ids in enumerate(tokenized["input_ids"]) if len(ids) <= max_context_window]
    tokenized["input_ids"] = [tokenized["input_ids"][i] for i in valid_indices]
    tokenized["attention_mask"] = [tokenized["attention_mask"][i] for i in valid_indices]
    
    return tokenized


def tokenize_query(query: str) -> str:
    max_seq_len = 350 # llama3-8b-instruct official context window
    max_context_window = 8000
    eval_ds = Dataset.from_dict(query)
    eval_text = eval_ds.map(
    create_inference_text,
    batched=True, num_proc=1,
    remove_columns=eval_ds.column_names,
    )
    eval_tokenized = eval_text.map(
        tokenize_and_truncate,
        batched=True, num_proc=1,
        remove_columns=eval_text.column_names,
        fn_kwargs={"max_context_window": max_context_window, "max_seq_len": max_seq_len, "mode": "eval"}
    )

    return eval_tokenized


def run_model_inference(model, tokenizer, tokenized_query: str) -> str:
    """Runs model inference"""
    ids = torch.tensor(tokenized_query['input_ids']).to(model.device)
    masks = torch.tensor(tokenized_query['attention_mask']).to(model.device)
    model_inputs = {"ids": ids, "masks": masks}
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts

@mcp.tool()
def run_arxiv_search_pipeline(query: str) -> str:
    """Runs the arxiv search pipeline"""
    instruction_path = "../instruction.txt"

    ############### temporary override: load the query in from a file
    data_path = "../single_query.jsonl"
    query = load_data(data_path, instruction_path)
    model_output = query["output"][0]
    ###############

    ############### run model inference - switched off until model deployment is figured out
    # json_query = prepare_query(query, instruction_path)
    # tokenized_query = tokenize_query(json_query)
    # model_output = run_model_inference(model, tokenizer, tokenized_query)
    ###############

    ############### parse the model output
    arxiv_query = markdown_to_json(model_output)
    ###############

    ############### request from arxiv
    arxiv_response = request_arxiv(arxiv_query)
    ###############

    ############### parse the arxiv response
    arxiv_response_parsed = parse_arxiv_response(arxiv_response)
    ###############

    return arxiv_response_parsed


if __name__ == "__main__":
    mcp.run()