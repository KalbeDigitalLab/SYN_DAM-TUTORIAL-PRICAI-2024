# please run `pip3 install datasets` before run script bellow

from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="/content/food102", split="train")
dataset.push_to_hub("adhisetiawan/food102")