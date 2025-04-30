# LLM-Powered Research Paper Discovery Tool

A lightweight tool that retrieves relevant research papers from Arxiv based on user queries, leveraging a fine-tuned open-source LLM and Cursor IDE integration via MCP.


<video width="70%" controls>
  <source src="demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Key Features

- ✅ **Lightweight**: Fine-tuned Meta Llama 3.2 1B Instruct model, which can be loaded unquantized on a 16GB Macbook Air.
- ✅ **Open Source**: Model available on 🤗. Simply use the model id "Shaikh58/llama-3.2-1b-instruct-lora-arxiv-query". Full model card available on [Hugging Face](https://huggingface.co/Shaikh58/llama-3.2-1b-instruct-lora-arxiv-query).  Papers retrieved from open source Arxiv API.
- ✅ **Cursor Integration**: Seamless workflow through MCP server integration. No need to leave the IDE or load the model yourself.
- ✅ **Custom Search**: Supports various search constraints and filtering options such as citation count and publication date


## User Guide
Clone the repository: 
```
git clone https://github.com/shaikh58/llm-paper-retriever.git
cd llm-paper-retriever
```
This project uses uv to manage dependencies. Install uv:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Install dependencies: 
```
uv sync
```
Add cursor mcp json file and cursor rules file to your project. Update the path in the mcp.json file with the path to this repository on your local machine. 
```
mkdir -p /your/cursor/project/.cursor
cp .cursor/mcp.json /your/cursor/project/.cursor/mcp.json
cp .cursorrules /your/cursor/project/.cursorrules
```

Check whether Cursor recognizes the server: From the Menu bar, go to 'Cursor' -> 'Settings' -> Cursor Settings -> MCP.\
If everything works, you should see the server in the list. You may need to refresh the list.

### Its ready to try out!

#### Note: This tool loads in a fine tuned 1B parameter model from the Hugging Face hub. It is small enough to fit unquantized on a 16GB M3 Macbook Air, and may even work with as low as 8GB of GPU memory (unified memory if you're a Mac user). 

Tools in Cursor only work in Agent mode. Switch to Agent mode in your chat window, and ask it to find you research papers! It should connect to this tool and ask 
for your permission to run it. Wait a few seconds and it should return a list of papers.

For best results, only enter your query in the chat window. Do not include any other text. e.g. "Find me papers on transformer architectures published in since 2023 with at least 100 citations".


## Deep dive: LLM Fine-tuning

- **Base model**: Meta Llama 3.2 1B Instruct
- **Structured Output**: The model is trained to output structured markdown instead of conversational text. This makes it possible to parse the output and construct a query for a search API.
- **Generate realistic training data**: Synthetically generated dataset designed to teach the model to output structured markdown. Added variance to the training data by passing generated user queries through an LLM to make the user queries more conversational and realistic.
- **Dataset size**: 10,000 synthetically generated queries enough to achieve satisfactory performance.
- **LoRA fine tuning**: Fine-tuning process uses LoRA (Low-Rank Adaptation) to efficiently adapt the base model for our specific structured output task while maintaining its general knowledge capabilities.


### Dataset Sample

| User Query | Final JSON output (after parsing) |
|------------|------------------------------|
| "Find recent papers on transformer architectures in NLP published in the last 2 years with at least 100 citations" | ```{"keyword": "transformers", "year": (2023, ">="), "sort_by": "year", "limit": 10}``` | 


## License

This project is released under the MIT License - see the LICENSE file for details.