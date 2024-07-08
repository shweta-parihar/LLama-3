# FOR SERIAL OPERATION ON GPU:

1. Run below command from terminal CLI.

      $ pip install -r requirements.txt

2. Run "llama_3_text_gen.py" file
   
      $ python llama_3_text_gen.py
   
3. After running above file, results will be saved in file "df_results.parquet" in the current directory.


# FOR PARALLEL OPERATION ON MULTIPLE GPUs:

1. Run below command from terminal CLI.

      $ pip install -r requirements.txt

2. Run "parallel_processing_llama_text_gen.py" file
   
      $ python parallel_processing_llama_text_gen.py
   
3. After running above file, results will be saved in file "df_results_parallel.parquet" in the current directory.
