import streamlit
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_PATH = 'model/'
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

streamlit.title('Light OCR')

uploaded_file = streamlit.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    streamlit.image(image, caption='Uploaded Image', use_container_width=True)
    
    pixel_values = processor(images=image, return_tensors='pt').pixel_values
    output_ids = model.generate(pixel_values)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    streamlit.subheader('Recognized Text')
    streamlit.write(text)

