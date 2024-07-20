## FOR SERIAL OPERATION ON GPU:

1. Run below command from terminal CLI.

      $ pip install -r requirements.txt

2. Run "llama_3_text_gen.py" file
   
      $ python llama_3_text_gen.py
   
3. After running above file, results will be saved in file "df_results.parquet" in the current directory.


## FOR PARALLEL OPERATION ON MULTIPLE GPUs:

### For LLM Inference
1. Run below command from terminal CLI.

      $ pip install -r requirements.txt

2. Run "parallel_processing_LLM.py" file
   
      $ python parallel_processing_LLM.py
   
3. After running above file, results will be saved in multiple parquet files in the current directory. Eg. df_10k_LLM_results_1, df_10k_LLM_results_2 and so on. Number of files generated for results will be equal to the number of GPUs used.

### For RAG Inference
1. Run below command from terminal CLI.

      $ pip install -r requirements.txt

2. Run "parallel_processing_RAG.py" file
   
      $ python parallel_processing_RAG.py
   
3. After running above file, results will be saved in multiple parquet files in the current directory. Eg. df_10k_RAG_results_1, df_10k_RAG_results_2 and so on. Number of files generated for results will be equal to the number of GPUs used.
