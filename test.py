import os

cache_directory = os.path.expanduser("~/.cache/huggingface/modules/datasets_modules")

def print_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        if files:
            print(f"Files in directory '{directory}':")
            for file_name in files:
                print(file_name)
        else:
            print(f"No files found in directory '{directory}'.")
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")

# Print files in the cache directory
print_files_in_directory(cache_directory)