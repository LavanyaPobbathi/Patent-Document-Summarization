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

import requests
import pandas as pd

# Read the first Excel sheet
df1 = pd.read_excel('Abstract_chunks_Summary_t5_small.xlsx')
df1 = df1.head(50)

# Read the second Excel sheet
df2 = pd.read_excel('Claims_chunks_Summary_t5_small.xlsx')
df2 = df2.head(50)

# Combine the desired columns from both dataframes into one variable
combined_text = df1['Abstract_chunks_Summary_t5_small'] + ' ' + df2['Claims_chunks_Summary_t5_small']

# Create a new dataframe with the combined text and other columns
output_df = pd.DataFrame({
    'Filename': df1['Filename'],
    'Abstract': df1['Abstract'],
    'Claims': df1['Claims'],
    'Abstract_chunks_Summary_t5_small': df1['Abstract_chunks_Summary_t5_small'],
    'Claims_chunks_Summary_t5_small': df2['Claims_chunks_Summary_t5_small'],
})


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
for combined in tqdm(combined_text, desc="Generating summaries", unit="claim"):
    generated_text = generate_text_chunks(combined, tokenizer, model)
    generated_texts.append(generated_text)


# Add the summaries to the data frame
output_df['Combined_Summary_chunks_t5_small'] = generated_texts

# Save the data frame to a new Excel file
output_file = 'Combined_Google_patent_chunks_Summary_t5_small.xlsx'
output_df.to_excel(output_file, index=False)

