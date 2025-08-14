import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from config import MODEL_NAME

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

model.config.pad_token_id = processor.tokenizer.pad_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = processor.tokenizer.eos_token_id
