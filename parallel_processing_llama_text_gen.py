import multiprocessing
import numpy as np
import transformers
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from time import time
from tqdm import tqdm
import pandas as pd


def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds
    

# Define the function to generate the response
def generate_response(question, pipeline):
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

def llm_process(df, gpu_device, output):

    time_1 = time()
       
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
    
    text_gen_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    # if torch.cuda.is_available():
    #     gpu_count = torch.cuda.device_count()
    #     for i in range(gpu_count):
    #         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    #         print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
    #         print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
    #         print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")

    
    
    # Apply the function to each row in the DataFrame and save the results in a new column
    
    tqdm.pandas()  # Enable the progress bar for pandas apply
    
    
    
    # df = pd.read_parquet("df_short.parquet", engine='fastparquet')  
        
    df['llm_answer'] = df['question_text'].progress_apply(lambda x: generate_response(x, text_gen_pipeline))


    time_2 = time()
    
    hours, minutes, seconds = format_time(int(time_2 - time_1))
    # output.put('Process ' + str(gpu_device) + ' finished in {hours} hours, {minutes} minutes, {seconds} seconds')
    print(f'\n------------------------------------------------------------------------\nProcess {gpu_device} finished in  ==>  {hours} hours, {minutes} mins, {seconds} secs.\nWaiting for other processes to finish ...\n------------------------------------------------------------------------\n')

    output.put(df)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Set start method to 'spawn'

    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
    else:
        raise RuntimeError("No GPUs available!")

    
    df = pd.read_parquet("df_short.parquet", engine='fastparquet')  
    df_list = np.array_split(df, num_gpus)

    # Create a queue to collect results
    output = multiprocessing.Queue()
  
    process_list = []
    for i in range(len(df_list)):    
        # Create processes for each list
        process_list.append(multiprocessing.Process(target=llm_process, args=(df_list[i], i, output)))
    
    time_3 = time()
    for process in process_list:
        # Start the processes
        process.start()
    
    for process in process_list:
        # Wait for both processes to finish
        process.join()

    time_4 = time()

    # for i in range(len(process_list)):
    #     # Retrieve the results from the queue
    #     print(output.get())

    
    # Collect the results from the queue
    df_results_list = [output.get() for _ in range(num_gpus)]

    # Combine the dataframes
    df_results = pd.concat(df_results_list, ignore_index=True)

    # Save the dataframe with results
    df_results.to_parquet('df_results_parallel.parquet', engine='fastparquet', compression='gzip')

    hours, minutes, seconds = format_time(int(time_4 - time_3))
    
    # Print the results
    print(f"\n------------------------------------------------------------------------\nAll processes finished!\nTotal time  ==>  {hours} hours, {minutes} mins, {seconds} secs.\n------------------------------------------------------------------------\n")

