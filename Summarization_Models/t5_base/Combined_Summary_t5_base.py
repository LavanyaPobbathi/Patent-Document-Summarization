import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# Fixing the random seed
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)

# CUDA option
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
model.to(device)


import requests
import pandas as pd

# Read the first Excel sheet
df1 = pd.read_excel('Abstract_Summary_t5_base.xlsx')
df1 = df1.head(50)

# Read the second Excel sheet
df2 = pd.read_excel('Claims_Summary_t5_base.xlsx')
df2 = df2.head(50)

# Combine the desired columns from both dataframes into one variable
combined_text = df1['Abstract_Summary_t5_base'] + ' ' + df2['Claims_Summary_t5_base']

# Create a new dataframe with the combined text and other columns
output_df = pd.DataFrame({
    'Filename': df1['Filename'],
    'Abstract': df1['Abstract'],
    'Claims': df1['Claims'],
    'Abstract_Summary_t5_base': df1['Abstract_Summary_t5_base'],
    'Claims_Summary_t5_base': df2['Claims_Summary_t5_base'],
})

# List to store the generated text
generated_texts = []

# Generate text for each abstract
for combined in tqdm(combined_text, desc="Generating summaries", unit="input"):
    inputs = tokenizer.encode("summarize: " + combined, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts.append(generated_text)


# Add the summaries to the data frame
output_df['Combined_Summary'] = generated_texts

# Save the data frame to a new Excel file
output_file = 'Combined_Google_patent_Summary_t5_base.xlsx'
output_df.to_excel(output_file, index=False)