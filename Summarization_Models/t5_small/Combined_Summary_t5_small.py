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
model_name_or_path = 'turingmachine/hupd-t5-small'
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
# Model
model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
model.to(device)

import requests
import pandas as pd

# Read the first Excel sheet
df1 = pd.read_excel('abstract_summary_t5-small.xlsx')
df1 = df1.head(50)

# Read the second Excel sheet
df2 = pd.read_excel('claim_summary_t5-small.xlsx')
df2 = df2.head(50)

# Combine the desired columns from both dataframes into one variable
combined_text = df1['Abstract_Summary'] + ' ' + df2['claim_Summary']

# Create a new dataframe with the combined text and other columns
output_df = pd.DataFrame({
    'Filename': df1['Filename'],
    'Abstract': df1['Abstract'],
    'Claims': df1['Claims'],
    'Abstract_Summary': df1['Abstract_Summary'],
    'claim_Summary': df2['claim_Summary'],
})

"""
# Load the Excel file
df = pd.read_excel('/content/drive/MyDrive/All_Patent.xlsx')
df = df.head(50)
# Extract the desired columns
columns_to_keep = ['Filename', 'Abstract', 'Claims']  # Add other column names here
df_subset = df[columns_to_keep]

# Store the 'Abstract' and 'Claims' columns in one variable
abstracts = df['Abstract']
"""
# List to store the generated text
generated_texts = []

# Generate text for each abstract
for combined in tqdm(combined_text, desc="Generating summaries", unit="input"):
    inputs = tokenizer(combined, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=512, num_return_sequences=1, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts.append(generated_text)


# Add the summaries to the data frame
output_df['Combined_Summary'] = generated_texts

# Save the data frame to a new Excel file
output_file = 'Google_patent_Summary_t5_small.xlsx'
output_df.to_excel(output_file, index=False)