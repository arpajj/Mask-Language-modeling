# Mask-Language-Modeling for NLI

This is a project where we apply Mask Language Modeling (MLM) after using a prompting technique 
in order to perform Natural Language Inference (NLI). The models used are BERT, RoBERTa, GPT2 and BART.

Create templates by using `python create_templates.py`.

To use GPT-2 model, run `python run_gpt2.py --model MODEL_NAME --dataset DATASET_NAME --template START`.

To use BERT, RoBERTa and BART model, run `python runner.py --model MODEL_NAME --dataset DATASET_NAME --template START`.