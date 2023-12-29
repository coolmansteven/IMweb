# Artificial Intelligence Club Project: Image Captioning with Retrained Transformer Model

## Description

This project focuses on retraining a transformer model for image captioning, specifically tailored for recognizing and describing people in various actions.

## Files

### `Clean_Image_Captioning.ipynb`

This Jupyter Notebook serves the purpose of retraining the base model. It offers flexibility to the user by allowing them to subset data and save the model in any desired format. Additionally, there is an option to push the model to Hugging Face or create a PyTorch save on a local device.

### `Main.py`

This Python script is designed to be run on a local device. It imports our team's saved version of the model and processor from Hugging Face. The script uses Streamlit to create a user interface for easy image upload and displays generated captions for the uploaded images.

**Usage Instructions:**
1. After running the file, launch the application in the terminal by typing: `streamlit run main.py`

## How to Contribute

Feel free to contribute to the project by forking the repository, making your changes, and creating a pull request. We welcome improvements, bug fixes, and new features.

## Dependencies

Make sure to install the required dependencies listed in the `requirements.txt` file before running the code. (May be more required) 

## Contact

If you have any questions or suggestions, please feel free to reach out to us at [Strujillo6686@sdsu.edu].
