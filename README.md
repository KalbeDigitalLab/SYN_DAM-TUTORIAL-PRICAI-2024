# SYN_DAM-TUTORIAL-PRICAI-2024

This repository contains materials and code for the **PRICAI 2024** tutorial on synthetic data generation using diffusion models. In this project, we demonstrate how to generate, integrate, and utilize synthetic images to expand and improve an image classification dataset, specifically targeting Indonesian food items. The tutorial will guide you through generating synthetic images, adding them to an existing dataset, training a classifier, and fine-tuning the model if necessary for better results.

## Project Overview

In this tutorial, we use **Stable Diffusion XL (SDXL)** along with the **Hugging Face Diffusers** and **Transformers** libraries to generate and fine-tune images, aiming to add a new class to the **Food101** dataset. This workflow can be valuable in situations where labeled data is scarce or costly to obtain.

### Tutorial Workflow

1. **Generate Synthetic Images (1000 samples)**  
   We start by generating 1000 synthetic images representing Indonesian food items using SDXL. This step involves setting up SDXL in **Diffusers** and using targeted prompts to create realistic and varied images.

2. **Fine-Tune SDXL for Higher Quality Images**  
   If the generated images don’t meet the desired quality, we’ll demonstrate how to fine-tune SDXL on a smaller, curated dataset to achieve higher fidelity. This refinement can lead to more accurate and realistic image generation for specific food items.

3. **Add Synthetic Images as a New Class in Food101 Dataset**  
   Next, we integrate the generated images into the Food101 dataset by creating a new class, making the dataset more diverse and inclusive of specific regional cuisine.

4. **Train and Test a Classification Model**  
   Using the extended Food101 dataset, we train a classification model to test how well it can recognize both synthetic and real images. This step will highlight the impact of synthetic data on model performance, particularly when applied to real-world images.



## Prerequisites

1. **Clone the Repository and Set Up Environment**

   Start by cloning this repository and creating a virtual environment:

   ```bash
   git clone https://github.com/yourusername/indonesian-food-synthetic-data.git
   cd indonesian-food-synthetic-data
   python -m venv env
   source env/bin/activate  # For Windows, use `env\Scripts\activate`
   ```

2. **Install Dependencies**

   Install all required libraries from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the following libraries included in `requirements.txt`:

   ```plaintext
   torch
   diffusers
   transformers
   datasets
   wandb
   ```

## Project Structure

```plaintext
.
├── Diffusion_Generate_Data.ipynb         # Notebook for generating synthetic data using diffusion model
├── Fine_tune_Diffusion_Model_on_Specific_Style_image.ipynb # Notebook for fine-tuning SDXL on specific styles
├── image_classification.ipynb            # Notebook for training and testing the classification model
├── push_dataset_to_hub.py                # Script to push the dataset to the Hugging Face Hub
└── README.md                             # Project documentation
```

## Usage

### 1. Generate Synthetic Images

Open the `Diffusion_Generate_Data.ipynb` notebook. This notebook will guide you through generating 1000 synthetic images of Indonesian food using a diffusion model (SDXL). Follow the prompts in the notebook to generate and save the images.

### 2. Fine-Tune Diffusion Model on Specific Styles (Optional)

If the initial synthetic images lack the desired quality or specificity, open `Fine_tune_Diffusion_Model_on_Specific_Style_image.ipynb` to fine-tune SDXL on a small set of curated images. This step will help produce higher-quality images for specific food items.

### 3. Train a Classification Model

After generating and preparing the synthetic images, open the `image_classification.ipynb` notebook. This notebook demonstrates how to integrate the new synthetic images into the Food101 dataset and train a classifier. The trained model will then be evaluated to see how well it generalizes to real-world images.

### 4. Push Dataset to Hugging Face Hub

To share the extended dataset (with synthetic data added) on the Hugging Face Hub, use the `push_dataset_to_hub.py` script:

```bash
python push_dataset_to_hub.py --dataset_dir path/to/extended_food101 --dataset_name your_username/extended_food101
```

Replace `path/to/extended_food101` with the actual path to your dataset, and `your_username/extended_food101` with your preferred name on the Hugging Face Hub.


## Results and Evaluation

After training the classifier, we can assess how well it performs on real images within the Food101 dataset. This evaluation demonstrates the effectiveness of synthetic data in real-world applications, especially for cases where labeled data is limited.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License.