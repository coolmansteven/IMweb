import streamlit as st
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

# Load my_model from the current directory on your local device
model = torch.load('/Users/stevotrujillo/Desktop/Image Captioning/my_model')
# Put the model into evaluation mode
model.eval()
# Load the processor (working on saving my own processor)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Title
st.title("AI Club Image Captioning Model")

# Header
st.header("Image Upload:")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    image = Image.open(uploaded_file)
    inputs = processor(images=image, return_tensors="pt").to('cpu')
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader('Generated Image Caption:')
    st.write(generated_caption)






