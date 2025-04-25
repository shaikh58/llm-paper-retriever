import random
import json
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Structure to group related fields, keywords, and journals
THEMES = [
    {
        "theme_name": "AI/ML",
        "fields": ["machine learning", "artificial intelligence", "natural language processing", "computer vision", "robotics"],
        "keywords": ["transformer models", "neural networks", "reinforcement learning", "generative models", "explainable AI", "deep learning", "computer vision"],
        "journals": ["Journal of Machine Learning Research", "NeurIPS Proceedings", "ICLR Proceedings", "CVPR Proceedings", "Nature Machine Intelligence", "IEEE Transactions on Pattern Analysis and Machine Intelligence", "JMLR", "Transactions on Machine Learning Research"]
    },
    {
        "theme_name": "Physics/Astronomy",
        "fields": ["astrophysics", "particle physics", "cosmology", "condensed matter physics", "quantum mechanics"],
        "keywords": ["dark matter", "black holes", "exoplanets", "gravitational waves", "standard model", "quantum field theory", "superconductivity", "quantum entanglement"],
        "journals": ["Physical Review Letters", "Nature Astronomy", "The Astrophysical Journal", "Monthly Notices of the Royal Astronomical Society", "Journal of Cosmology and Astroparticle Physics", "Physical Review B", "Nature Physics"]
    },
    {
        "theme_name": "Biology/Genetics",
        "fields": ["genetics", "bioinformatics", "molecular biology", "cell biology", "evolutionary biology"],
        "keywords": ["CRISPR", "gene editing", "DNA sequencing", "genomics", "proteomics", "transcriptomics", "synthetic biology", "epigenetics"],
        "journals": ["Nature Genetics", "Cell", "Science", "PLOS Genetics", "Genome Biology", "Nucleic Acids Research", "Molecular Cell", "eLife"]
    },
    {
        "theme_name": "Medicine/Health",
        "fields": ["oncology", "cardiology", "immunology", "public health", "epidemiology", "virology", "pharmacology", "neurology"],
        "keywords": ["vaccine development", "clinical trials", "cancer research", "cardiovascular disease", "immune response", "pandemic modeling", "drug discovery", "neurodegenerative diseases"],
        "journals": ["Nature Medicine", "The Lancet", "JAMA", "NEJM", "Cell Host & Microbe", "Journal of Virology", "American Journal of Epidemiology", "British Medical Journal", "Annals of Internal Medicine"]
    },
    {
        "theme_name": "Chemistry/Materials",
        "fields": ["organic chemistry", "physical chemistry", "biochemistry", "inorganic chemistry", "materials science", "nanotechnology", "polymer science"],
        "keywords": ["catalysis", "synthesis", "spectroscopy", "protein folding", "graphene", "nanoparticles", "self-assembly", "electrochemistry"],
        "journals": ["JACS", "Angewandte Chemie", "Nature Chemistry", "Chemical Science", "ACS Nano", "Journal of Physical Chemistry", "Nature Materials", "Advanced Materials"]
    },
    {
        "theme_name": "Economics/Finance",
        "fields": ["macroeconomics", "microeconomics", "econometrics", "behavioral economics", "finance", "development economics"],
        "keywords": ["game theory", "market analysis", "inflation", "monetary policy", "risk management", "asset pricing", "inequality", "behavioral finance"],
        "journals": ["American Economic Review", "Quarterly Journal of Economics", "Journal of Political Economy", "Econometrica", "Journal of Finance", "Review of Economic Studies", "Journal of Financial Economics"]
    },
    {
        "theme_name": "Environmental Science",
        "fields": ["climate change", "conservation biology", "pollution control", "renewable energy", "oceanography", "ecology", "atmospheric science"],
        "keywords": ["carbon capture", "biodiversity", "sustainable development", "climate modeling", "plastic pollution", "solar energy", "ecosystem dynamics", "deforestation"],
        "journals": ["Nature Climate Change", "Environmental Science & Technology", "Journal of Climate", "Global Change Biology", "Conservation Letters", "Science Advances", "Nature Sustainability"]
    },
    {
        "theme_name": "Psychology/Neuroscience",
        "fields": ["psychology", "neuroscience", "cognitive science", "behavioral science"],
        "keywords": ["cognitive bias", "decision making", "brain imaging", "neural circuits", "memory formation", "mental health", "social cognition", "perception"],
        "journals": ["Psychological Review", "Nature Neuroscience", "Journal of Neuroscience", "Trends in Cognitive Sciences", "Neuron", "Psychological Science", "Cognition"]
    },
     {
        "theme_name": "Engineering",
        "fields": ["civil engineering", "mechanical engineering", "electrical engineering", "chemical engineering", "aerospace engineering", "biomedical engineering"],
        "keywords": ["fluid dynamics", "structural analysis", "control systems", "semiconductors", "robotics", "thermodynamics", "signal processing", "tissue engineering"],
        "journals": ["IEEE Transactions on Automatic Control", "Journal of Fluid Mechanics", "Nature Biomedical Engineering", "Advanced Functional Materials", "Journal of the Mechanics and Physics of Solids", "Chemical Engineering Journal"]
    },
    {
        "theme_name": "Humanities/Social Sciences",
        "fields": ["sociology", "political science", "anthropology", "archaeology", "history", "philosophy", "linguistics"],
        "keywords": ["social networks", "political theory", "cultural evolution", "historical analysis", "ethics", "language acquisition", "archaeological methods", "governance"],
        "journals": ["American Journal of Sociology", "American Political Science Review", "Current Anthropology", "Journal of Archaeological Science", "The American Historical Review", "Mind", "Language", "Journal of Social History"]
    }
]

# Authors list remains generic as authors can span related fields
AUTHORS = [
    "Yoshua Bengio", "Geoffrey Hinton", "Yann LeCun", "Jennifer Doudna", "Emmanuelle Charpentier",
    "Shinya Yamanaka", "Kip Thorne", "Rainer Weiss", "Barry Barish", "Frances Arnold",
    "George Smith", "Gregory Winter", "James Allison", "Tasuku Honjo", "Michael Rosbash",
    "Jeffrey Hall", "Michael Young", "Richard Henderson", "Joachim Frank", "Jacques Dubochet",
    "Jean-Pierre Sauvage", "Fraser Stoddart", "Ben Feringa", "Paul Modrich", "Aziz Sancar",
    "Tomas Lindahl", "Eric Betzig", "Stefan Hell", "William Moerner", "Ada Yonath", "Venki Ramakrishnan",
    "Thomas Steitz", "Elizabeth Blackburn", "Carol Greider", "Jack Szostak", "Peter Higgs", "François Englert",
    "Daniel Kahneman", "Elinor Ostrom", "Paul Krugman", "Noam Chomsky", "Jane Goodall", "Stephen Hawking",
    "Judith Butler", "Max Weber", "Karl Marx", "Marie Curie", "Albert Einstein", "Isaac Newton", "Charles Darwin"
]



# --- Helper Functions ---

def generate_year_constraint():
    """Generates a random year constraint."""
    type = random.choice(['>', '<', '>=', '<=', '=','between'])
    min_year = 2000
    max_year = 2024
    year = random.randint(min_year, max_year)
    if type == '=':
         return f"= {year}", f"in {year}"
    elif type == '>':
        return f"> {year}", f"after {year}"
    elif type == '<':
        return f"< {year}", f"before {year}"
    elif type == '>=':
        return f">= {year}", f"since {year}"
    elif type == '<=':
        return f"<= {year}", f"up to {year}"
    elif type == 'between':
        year2 = random.randint(year, max_year)
        return f">= {year}, <= {year2}", f"between {year} and {year2}"

def generate_citation_constraint():
    """Generates a random citation constraint."""
    type = random.choice(['>', '<', '>=', '<='])
    citations = random.choice([10, 20, 50, 100, 200, 500])
    if type == '>':
        return f"> {citations}", f"with more than {citations} citations"
    elif type == '<':
        return f"< {citations}", f"with fewer than {citations} citations"
    elif type == '>=':
        return f">= {citations}", f"with at least {citations} citations"
    elif type == '<=':
        return f"<= {citations}", f"with no more than {citations} citations"

def generate_query_and_output():
    """Generates a single input query and its corresponding structured output, ensuring thematic coherence."""
    # 1. Select a Theme
    theme = random.choice(THEMES)

    # 2. Select Field, Keywords, Journal from the chosen Theme
    topic = random.choice(theme["fields"])
    possible_keywords = theme["keywords"]
    possible_journals = theme["journals"]

    constraints_md = []
    constraints_query_parts = []
    options_md = {
        "Limit": 10,
        "Sort By": "relevance",
        "Sort Order": "descending"
    }
    input_prefix = ["Find papers about", "Show me papers about", "Search for papers about", "Give me papers on", "Show me papers on", "I want papers on", "Look for papers on"]
    query_parts = [f"{random.choice(input_prefix)} {topic}"] # Start with the basic query

    # --- Add Constraints (1 to 3 constraints, now theme-aware) ---
    num_constraints = random.randint(1, 5)
    added_constraint_types = set()

    # Prioritize theme-specific constraints first
    potential_constraint_types = ["year", "citations", "author", "keyword", "journal"]

    for _ in range(num_constraints):
        if not potential_constraint_types: # Should not happen with current setup, but safe check
            break

        constraint_type = random.choice(potential_constraint_types)

        # Avoid duplicate constraint types unless it's 'year' (e.g., >= 2018 and <= 2020)
        # For simplicity, we'll just prevent duplicates for now.
        if constraint_type in added_constraint_types:
             continue
        added_constraint_types.add(constraint_type)
        # potential_constraint_types.remove(constraint_type) # Remove to avoid duplicates

        if constraint_type == "year":
            value_md, value_query = generate_year_constraint()
            constraints_md.append(f"- **Year**: {value_md}")
            constraints_query_parts.append(value_query)
        elif constraint_type == "citations":
            value_md, value_query = generate_citation_constraint()
            constraints_md.append(f"- **Citations**: {value_md}")
            constraints_query_parts.append(value_query)
        elif constraint_type == "author":
            author = random.choice(AUTHORS) # Author choice remains broad
            constraints_md.append(f"- **Author**: {author}")
            constraints_query_parts.append(f"by {author}")
        elif constraint_type == "journal":
            journal = random.choice(possible_journals)
            constraints_md.append(f"- **Journal**: {journal}")
            constraints_query_parts.append(f"in the journal '{journal}'")
        elif constraint_type == "keyword":
            keyword = random.choice(possible_keywords)
            # Avoid keyword being identical to the main topic
            if keyword.lower() != topic.lower():
                constraints_md.append(f"- **Keyword**: {keyword}")
                constraints_query_parts.append(f"mentioning '{keyword}'")
            else:
                 added_constraint_types.remove(constraint_type) # Remove if keyword is same as topic

    # --- Add Sorting (optional override) ---
    if random.random() < 0.2: # 30% chance to specify sorting
        sort_choice = random.choice(["date", "citations", "relevance"])
        options_md["Sort By"] = sort_choice
        if sort_choice == "date":
            # Ensure "sorted by most recent" isn't added redundantly if already present
            if "sorted by most recent" not in constraints_query_parts:
                 constraints_query_parts.append("sorted by most recent")
            options_md["Sort Order"] = "descending"
        elif sort_choice == "citations":
             if "sorted by citation count" not in constraints_query_parts:
                 constraints_query_parts.append("sorted by citation count")
             options_md["Sort Order"] = "descending"
        elif sort_choice == "relevance":
             # Add explicit relevance sort phrase only if other constraints exist and it wasn't added
             if constraints_query_parts and "sorted by relevance" not in constraints_query_parts:
                 constraints_query_parts.append("sorted by relevance")
             elif not constraints_query_parts: # If no other constraints, just ask for papers (implies relevance)
                 query_parts[0] = f"Show me the most relevant papers on {topic}"
             options_md["Sort Order"] = "descending" # Default relevance sort is descending


    # --- Construct Input Query ---
    random.shuffle(constraints_query_parts) # Mix the order of constraints
    input_query = query_parts[0]
    if constraints_query_parts:
        # Join parts carefully, ensuring spaces but handling punctuation if needed
        query_suffix = " ".join(constraints_query_parts)
        input_query += " " + query_suffix
    # Add punctuation if the query doesn't end with it
    if not input_query.endswith(('.', '?', '!')):
        input_query += "."


    # --- Construct Output Markdown ---
    output_md = "## QUERY PARAMETERS\n\n"
    output_md += f"- **Topic**: {topic}\n\n"

    output_md += "## CONSTRAINTS\n\n"
    if constraints_md:
        # Sort constraints alphabetically by key for consistency
        constraints_md.sort(key=lambda x: x.split('**')[1])
        output_md += "\n".join(constraints_md) + "\n\n"
    else:
        output_md += "\n" # Ensure a blank line even if no constraints

    output_md += "## OPTIONS\n\n"
    output_md += f"- **Limit**: {options_md['Limit']}\n"
    output_md += f"- **Sort By**: {options_md['Sort By']}\n"
    output_md += f"- **Sort Order**: {options_md['Sort Order']}\n"

    return {"input": input_query, "output": output_md.strip()}



model = AutoModelForCausalLM.from_pretrained(
    # "meta-llama/Meta-Llama-3-8B-Instruct", # only when downloading for the first time
    "./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct",
    # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto"
 )
tokenizer = AutoTokenizer.from_pretrained("./hf_cache/meta-llama/Meta-Llama-3-8B-Instruct-tokenizer")


if __name__ == "__main__":
    NUM_SAMPLES = 50000
    REWRITE_EVERY_N_SAMPLES = 10

    # --- Generate Data ---
    sys_instruct = "Rewrite the following query to be more natural and conversational and human like. The response format should be the following: 'Rewritten query: BOS <rewritten query> EOS'."

    with open('./hf_cache/datasets/train/train.jsonl', 'w', encoding='utf-8') as f:
        for i in range(NUM_SAMPLES):
            if i % 100 == 0: 
                with open("./hf_cache/datasets/train/train_data_generation_log.txt", "a") as log_file:
                    log_file.write(f"Processed row {i}\n")

            data_point = generate_query_and_output()

            # inject variance into the input query
            if i % REWRITE_EVERY_N_SAMPLES == 0:
                tokenized_query = tokenizer(sys_instruct + "\n" + data_point['input'], return_tensors="pt").to(model.device)
                generated_ids = model.generate(**tokenized_query, max_new_tokens=70, do_sample=False)
                # Remove the string 'Rewritten query:' from the output by skipping the first few generated tokens. This will not deal with tokens added to the end.
                generated_text = tokenizer.batch_decode(generated_ids[:, len(tokenized_query['input_ids'][0]):], skip_special_tokens=False)[0]
                # remove any tokens after the string EOS is detected. Skip any that give errors.
                try:
                    generated_text = generated_text.split('EOS')[0].split('BOS')[1]
                    data_point['input'] = generated_text
                except:
                    with open("./hf_cache/datasets/train/train_data_generation_log.txt", "a") as log_file:
                        log_file.write(f"Error in generated text ----- {i} ----- text: {generated_text}\n")
                    print("Error in generated text")

            f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
            # Flush to ensure data is written immediately
            f.flush()