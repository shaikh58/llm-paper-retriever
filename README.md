# LLM-Powered Research Paper Discovery Tool

A lightweight tool that retrieves relevant research papers from Arxiv based on user queries, leveraging a fine-tuned open-source LLM and Cursor IDE integration via MCP.


<video width="70%" controls>
  <source src="demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Key Features

- ✅ **Open-Source**: Fine-tuned the small Meta Llama 3 8B Instruct model requiring only 40GB of GPU memory, rather than relying on proprietary APIs. Fine tuned model available on 🤗.
- ✅ **Cursor Integration**: Seamless workflow through MCP server integration
- ✅ **Custom Search**: Supports various search constraints and filtering options

## Architecture

![Architecture Diagram](architecture_diagram.png)

*The system architecture showing the flow from user input in Cursor IDE through the MCP server to the fine-tuned Llama 3 model, which generates structured markdown that gets parsed to JSON for querying the arXiv API.*


## LLM Fine-tuning

In this project, I demonstrate a simple example of how to fine-tune an open-source LLM for specialized tasks. I choose a model small enough to fit on e.g. RTX6000 or a single A40 GPU.

- **Base model**: Meta Llama 3 8B Instruct
- **Structured Output**: The model is trained to output structured markdown instead of conversational text. This makes it possible to parse the output and construct a query to a search engine.
- **Generate training Data with added variance**: Synthetically generated dataset designed to teach the model to output structured markdown. Added variance to the training data by passing generated user queries to an LLM to make the user queries more conversational and realistic
- **LoRA fine tuning**: Fine-tuning process uses LoRA (Low-Rank Adaptation) to efficiently adapt the base model for our specific structured output task while maintaining its general knowledge capabilities.


## MCP Server Integration with Cursor IDE

I setup a basic MCP server to allow the model to be used as a tool within the Cursor environment

- Enables seamless workflow from query to results without leaving the IDE


## Results

### Training Metrics

![Training and Validation Loss Curves](training_loss_curve.png)

*Caption: Training and validation loss over 10 epochs showing convergence with minimal overfitting.*

### Dataset Examples

| User Query | Generated Structured Markdown |
|------------|------------------------------|
| "Find recent papers on transformer architectures in NLP published in the last 2 years" | ```json\n{"topic": "transformer architectures", "field": "NLP", "date_range": "2022-2024", "sort_by": "relevance"}\n``` |
| "Show me the most cited papers on protein folding from the top journals" | ```json\n{"topic": "protein folding", "sort_by": "citation_count", "journal_quality": "top", "limit": 10}\n``` |

### Fine-tuning Results

| Dataset Size | ROUGE-L | Exact Match 
|--------------|---------|-------------|
| 100  | 0.78    | 0.45        | 0.82              
| 500  | 0.85    | 0.62        | 0.91            
| 1000  | 0.89   | 0.70        | 0.95           


## Installation
Clone the repository: 
```
git clone https://github.com/shaikh58/paper-retriever-workflow.git
cd paper-retriever-workflow
```
Install dependencies: 
```
pip install -r requirements.txt
```
Download the fine-tuned model: 
```
model = AutoModelForCausalLM.from_pretrained("shaikh58/llama-3-8b-instruct-arxiv-search")
```
Start the server: 
```
python launch_mcp_server.py
```
## Usage

1. Install the companion extension in Cursor IDE
2. Connect to your local MCP server
3. Use the command palette to access the Research Paper Assistant
4. Enter your query with any constraints
5. Review and explore the returned papers

## License

This project is released under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta for releasing Llama 3 as an open-source model
- The Cursor IDE team for their MCP protocol
- arXiv for their open API