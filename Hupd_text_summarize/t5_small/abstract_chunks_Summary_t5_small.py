import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

# Fixing the random seed
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)

# CUDA option
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model/tokenizer name or path
model_name_or_path = 't5-small'
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
# Model
model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
model.to(device)

# Load the Excel file
df = pd.read_excel('All_Patent.xlsx')
df = df.head(50)

# Extract the desired columns
columns_to_keep = ['Filename', 'Abstract', 'Claims']  # Add other column names here
df_subset = df[columns_to_keep]

# Store the 'Claims' column in one variable
Abstracts = df['Abstract']

# List to store the generated text
generated_texts = []

def generate_text_chunks(input_text, tokenizer, model, chunk_size=512):
    # Tokenize the input text
    tokenized_text = tokenizer.encode(input_text)

    # Split the tokenized text into chunks
    text_chunks = [tokenized_text[i:i + chunk_size] for i in range(0, len(tokenized_text), chunk_size)]

    generated_text = ""
    for text_chunk in text_chunks:
        # Convert chunk to tensor and move to device
        inputs = torch.tensor([text_chunk]).to(device)

        # Generate text from the chunk
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=min(len(text_chunk) + 50, model.config.max_length))

        # Decode the output and add it to the generated text
        generated_text_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text += generated_text_chunk
    return generated_text

# Generate text for each claim
for Abstract in tqdm(Abstracts, desc="Generating summaries", unit="claim"):
    generated_text = generate_text_chunks(Abstract, tokenizer, model)
    generated_texts.append(generated_text)

# Add the summaries to the data frame
df_subset['Abstract_chunks_Summary_t5_small'] = generated_texts

# Save the data frame to a new Excel file
output_file = 'Abstract_chunks_Summary_t5_small.xlsx'
df_subset.to_excel(output_file, index=False)
