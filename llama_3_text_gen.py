

import transformers
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from time import time
from tqdm import tqdm
import pandas as pd

config_data = json.load(open("config.json"))

HF_TOKEN = config_data["HF_TOKEN"]
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          token=HF_TOKEN)
# tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    token=HF_TOKEN
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)


# Define the function to generate the response
def generate_response(question):
    messages = [
        {"role": "system", "content": "You are an honest and unbiased assistant who answers User queries with accurate responses."},
        {"role": "user", "content": question},
    ]
    
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        pad_token_id=pipeline.model.config.eos_token_id,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt):]

# Apply the function to each row in the DataFrame and save the results in a new column

tqdm.pandas()  # Enable the progress bar for pandas apply

time_1 = time()

df = pd.read_parquet("df_short.parquet", engine='fastparquet')  
    
df['llm_answer'] = df['question_text'].progress_apply(generate_response)

time_2 = time()
print("Time: ", time_2-time_1,"\n")

# Save the dataframe with results
df.to_parquet('df_results.parquet', engine='fastparquet', compression='gzip')