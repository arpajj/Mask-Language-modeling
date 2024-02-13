# Mask-Language-Modeling for NLI

This is a project where we apply Mask Language Modeling (MLM) after using a prompting technique 
in order to perform Natural Language Inference (NLI). The models used are BERT, RoBERTa, GPT2 and BART.

A)
Create templates by using `python create_templates.py`.

To use GPT-2 model, run `python run_gpt2.py --model MODEL_NAME --dataset DATASET_NAME --template START`.

To use BERT, RoBERTa and BART model, run `python runner.py --model MODEL_NAME --dataset DATASET_NAME --template START`.

B)
To run the analysis part of the paper go the folder "analysis".
Create templates by using "python create_templates.py" and entering the name of the dataset (sick, snli, multi_nli).

Make the dataloaders my running "make_dataloaders.py" and again the name of the dataset used. It also returns all the short and long dataloaders

Important tokens: For the extraction of tokens from the validation set run the "runner_valid.ipynb" and copy the tokens produced in the end
for the 3 labels. Then paste them into the "runner.ipynb" so to run the section of "Important Tokens" for the test set.

Short & Long sequences: After the run of "make_dataloaders.py" run the section "Short & Long Sequences" of the "runner.ipynb".
