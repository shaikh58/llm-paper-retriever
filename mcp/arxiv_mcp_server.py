from fastmcp import FastMCP
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM#, BitsAndBytesConfig
from peft import PeftModel
import re
from typing import Optional, Literal
from pydantic import BaseModel
import arxiv
from datetime import datetime
from requests import get

mcp = FastMCP("arxiv-mcp-server", dependencies=["transformers", "peft", "datasets", "pydantic", "torch", "typing", "json", "arxiv", "accelerate"])

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    trust_remote_code=True,
    # device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "Shaikh58/llama-3.2-3b-instruct-lora-arxiv-query"
)

tokenizer = AutoTokenizer.from_pretrained("Shaikh58/llama-3.2-1b-instruct-lora-arxiv-query",trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define the valid operators as string literals
ComparisonOperatorType = Literal[">", "<", ">=", "<=", "="]

class FilterCriteria(BaseModel):
    operator: ComparisonOperatorType
    value: int
    
    @classmethod
    def parse_filter_string(cls, filter_str: str) -> "FilterCriteria":
        # Remove any whitespace
        filter_str = filter_str.replace(" ", "")
        
        # Match patterns like >=50, >100, =75, etc.
        pattern = r'([><=]=?|<)(\d+)'
        match = re.match(pattern, filter_str)
        
        if match:
            operator_str, value_str = match.groups()
            # The operator_str is already in the correct format
            return cls(operator=operator_str, value=int(value_str))
        raise ValueError(f"Invalid filter criteria format: {filter_str}")

class Query(BaseModel):
    topic: Optional[str] = None
    journal: Optional[str] = None
    author: Optional[str] = None
    year: Optional[FilterCriteria] = None
    citations: Optional[FilterCriteria] = None
    keyword: Optional[str] = None
    limit: Optional[int] = 20
    # relevance not used for now
    sort_by: Optional[Literal["year", "citations", "relevance"]] = "citations"
    sort_order: Optional[Literal["ascending", "descending"]] = "descending"


def markdown_to_json(markdown_text: str) -> Query:
    """Converts markdown from model output to json for arxiv query"""
    # Initialize an empty dictionary to store our parameters
    params = {}
    # extract the "assistant" section of the model output
    assistant_section = markdown_text.split("<|assistant|>")[1]
    assistant_section = assistant_section.replace("*","")
    # Split the text into sections
    sections = assistant_section.split("##")
    for section in sections:
        if not section.strip():
            continue        
        lines = section.strip().split("\n")
        section_title = lines[0].strip().lower()
        
        # Process each bullet point in the section
        for line in lines[1:]:
            # Skip empty lines or lines without the bullet point format
            if not line.strip() or not line.strip().startswith("-"):
                continue
                
            # Remove the bullet point and any leading/trailing whitespace
            line = line.strip()[1:].strip()
            
            # Split on the first colon to get parameter name and value
            if ":" not in line:
                continue
                
            param_name, value = line.split(":", 1)
            param_name = param_name.strip().lower()
            value = value.strip()
            
            if param_name == "topic":
                params["topic"] = value
            elif param_name == "author":
                params["author"] = value
            elif param_name == "year":
                params["year"] = FilterCriteria.parse_filter_string(value)
            elif param_name == "journal":
                params["journal"] = value
            elif param_name == "citations":
                params["citations"] = FilterCriteria.parse_filter_string(value)
            elif param_name == "keyword":
                params["keyword"] = value
            elif param_name == "limit":
                params["limit"] = int(value)
            elif param_name == "sort by":
                params["sort_by"] = value.lower()
            elif param_name == "sort order":
                params["sort_order"] = value.lower()
    
    return Query(**params)

def to_arxiv_format(query: Query) -> str:
    """Converts json query to arxiv query format"""
    arxiv_query = ""
    attr_dict = query.dict()
    for param_name, value in attr_dict.items():
        if value is None:
            continue
        # can't use journal or topic because arxiv coding is difficult to map to; journal ref is not standardized
        # if param_name == "topic":
        #     arxiv_query += f"cat:%22{value}%22 AND "
        elif param_name == "author":
            arxiv_query += f"au:%22{value}%22 AND "
        # elif param_name == "journal":
        #     arxiv_query += f"jr:%22{value}%22 AND "
        # can't use citations because arxiv doesn't support it
        # elif param_name == "citations":
        #     ...
        elif param_name == "topic":
            arxiv_query += f"ti:%22{value}%22 AND "
        elif param_name == "keyword":
            arxiv_query += f"ti:%22{value}%22 AND "
        elif param_name == "year":
            date_now = datetime.now().strftime("%Y%m%d%H%M")
            date_earliest = "199101010000" # 1991 is the earliest year for arxiv
            operator = query.year.operator
            value = query.year.value
            if operator == ">" or operator == ">=":
                arxiv_query += f"submittedDate:[{value}01010000 TO {date_now}] AND "
            elif operator == "<" or operator == "<=":
                arxiv_query += f"submittedDate:[{date_earliest} TO {value}01010000] AND "
            elif operator == "=":
                arxiv_query += f"submittedDate:[{value}01010000 TO {value}12312359] AND "
                
    # remove the last AND
    arxiv_query = arxiv_query[:-5]

    return arxiv_query

def request_arxiv(arxiv_query: str, user_query: Query) -> str:
    """Requests arxiv papers"""
    client = arxiv.Client()
    search = arxiv.Search(
        query = arxiv_query,
        max_results = 100,
        sort_by = arxiv.SortCriterion.Relevance
    )
    results = client.results(search)
    all_results = list(results)
    if user_query.citations is None:
        return all_results
    
    citation_num = user_query.citations.value
    operator = user_query.citations.operator

    map_result_citations = {}
    filtered_results = []
    # get citation count from opencitations
    for result in all_results:
        # if no doi, discard it from the list
        if result.doi is None:
            continue
        API_CALL = f"https://opencitations.net/index/api/v2/citation-count/doi:{result.doi}"
        response = get(API_CALL)
        if response.status_code == 200:
            citation_count = int(response.json()[0]["count"])
            map_result_citations[result.doi] = citation_count
            if operator == ">" or operator == ">=":
                if citation_count > citation_num:
                    filtered_results.append(result)
            else:
                filtered_results.append(result)
        else:
            ...
    # sort the results by citations
    # check that map_result_citations is not empty
    if map_result_citations:
        filtered_results.sort(key=lambda x: map_result_citations[x.doi], reverse=True)
    else:
        ...

    return filtered_results[:user_query.limit]

def arxiv_to_chat(arxiv_response: list[arxiv.Result], arxiv_query: str) -> str:
    """Converts arxiv response to chat format"""
    if arxiv_response == []:
        output = "There were no results that matched your query and citation limit. If you specified a citation limit, try a lower number. If you did not specify a citation limit, try a different query."
        return output
    output = "Here are the papers I found that match your query. If you specified a citation limit, the results are filtered and sorted by citations: \n\n "

    for result in arxiv_response:
        for link in result.links:
            if link.title == "pdf":
                url = link.href
        list_authors = []
        for author in result.authors:
            list_authors.append(author.name)
        pub_date = result.published.strftime("%Y-%m-%d")
        update_date = result.updated.strftime("%Y-%m-%d")

        output += f"{result.title}\n"
        output += f"Authors: {', '.join(list_authors)}\n"
        # output += f"{result.summary}\n"
        output += f"URL: {url}\n"
        output += f"Published: {pub_date}\n"
        output += f"Updated: {update_date}\n"

    # output += f"\n\nThe query was: {arxiv_query}"

    return output

def load_data(data_path: str, instruction: str) -> dict:
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

def prepare_query(query: str, instruction: str) -> dict:
    """Convert string query into a JSON format for input to preprocessing pipeline"""
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


def preprocess_query(query: str) -> str:
    max_seq_len = 350 
    max_context_window = 8000 # llama3.2-1b-instruct official context window
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
        fn_kwargs={"max_context_window": max_context_window, "max_seq_len": max_seq_len}
    )

    return eval_tokenized


def run_model_inference(model, tokenizer, tokenized_query: str) -> str:
    """Runs model inference"""
    ids = torch.tensor(tokenized_query['input_ids']).to(model.device)
    masks = torch.tensor(tokenized_query['attention_mask']).to(model.device)
    generated_ids = model.generate(input_ids=ids, attention_mask=masks, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts


@mcp.tool()
def run_arxiv_search_pipeline(user_query: str = None) -> str:
    """Runs the arxiv search pipeline. Do not modify the user query."""

    instruction = """You are a specialized AI designed to parse research paper queries into a structured format. Always respond with a consistent markdown structure as shown below:

    ## QUERY PARAMETERS

    - **Topic**: [main research topic]

    ## CONSTRAINTS

    - **Year**: [operator] [value]
    - **Citations**: [operator] [value]
    - **Author**: [author names]
    - **Journal/Conference**: [publication venues]

    ## OPTIONS

    - **Limit**: [number of results]
    - **Sort By**: [date, citations]
    - **Sort Order**: [ascending, descending]

    Only include sections and parameters that are relevant to the query.
    Use operators like >, <, =, >=, <=, between, where appropriate.
    Do not provide explanations before or after the structured format.
    """

    ############### temporary override: simulate model output for dev purposes
    # model_output = "## QUERY PARAMETERS\n\n- **Topic**: machine learning\n\n## CONSTRAINTS\n\n- **Citations**: >= 10\n- **Keyword**: transformers\n- **Year**: > 2023\n\n## OPTIONS\n\n- **Limit**: 10\n- **Sort By**: relevance\n- **Sort Order**: descending"
    ###############

    ############### run model inference
    json_query = prepare_query(user_query, instruction)
    tokenized_query = preprocess_query(json_query)
    model_output = run_model_inference(model, tokenizer, tokenized_query)
    ###############

    ############### parse the model output
    query : Query = markdown_to_json(model_output[0])
    ###############

    ############### map to arxiv query format
    arxiv_query = to_arxiv_format(query)

    ############### request from arxiv
    arxiv_response = request_arxiv(arxiv_query, query)
    ###############

    ############### create a return message for the user
    output = arxiv_to_chat(arxiv_response, arxiv_query)
    ###############

    return output


if __name__ == "__main__":
    mcp.run()