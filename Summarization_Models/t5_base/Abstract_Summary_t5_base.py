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

# Load the Excel file
df = pd.read_excel('All_Patent.xlsx')
df = df.head(50)

# Extract the desired columns
columns_to_keep = ['Filename', 'Abstract', 'Claims']  # Add other column names here
df_subset = df[columns_to_keep]

# Store the 'Abstract' and 'Claims' columns in one variable
abstracts = df['Abstract']

# List to store the generated text
generated_texts = []

# Generate text for each abstract
for abstract in tqdm(abstracts, desc="Generating summaries", unit="abstract"):
    inputs = tokenizer.encode("summarize: " + abstract, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_texts.append(generated_text)

# Add the generated text to the DataFrame
df_subset['Abstract_Summary_t5_base'] = generated_texts

# Save the modified DataFrame to a new Excel file
df_subset.to_excel('Abstract_Summary_t5_base.xlsx', index=False)
