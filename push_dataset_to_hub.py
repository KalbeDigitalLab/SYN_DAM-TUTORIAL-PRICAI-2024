import argparse
from datasets import load_dataset

def push_dataset_to_hub(dataset_dir, dataset_name):
    # Load the dataset from the specified directory
    dataset = load_dataset("imagefolder", data_dir=dataset_dir, split="train")
    
    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(dataset_name)
    print(f"Dataset successfully pushed to hub with name: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to Hugging Face Hub")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for the Hugging Face Hub")

    args = parser.parse_args()

    # Push dataset to Hugging Face Hub using the provided arguments
    push_dataset_to_hub(args.dataset_dir, args.dataset_name)
