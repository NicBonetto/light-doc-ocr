import torch
from datasets import load_dataset
from config import DATASET_NAME, TRAIN_SPLIT, TEST_SPLIT_RATIO
from core.model import processor
from PIL import Image

def preprocess_batch(batch):
    images = [img.convert('RGB') for img in batch["image"]]

    labels = processor.tokenizer(batch['text'], padding=True, max_length=128, truncation=True).input_ids
    pixel_values = processor.image_processor(images, return_tensors="pt").pixel_values

    batch["pixel_values"] = pixel_values
    batch["labels"] = labels

    return batch

def load():
    dataset = load_dataset(DATASET_NAME, split = TRAIN_SPLIT)
    train_test = dataset.train_test_split(test_size = TEST_SPLIT_RATIO)
    train_ds = train_test['train']
    eval_ds = train_test['test']

    train_ds = train_ds.map(preprocess_batch, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(preprocess_batch, batched=True, remove_columns=eval_ds.column_names)

    return train_ds, eval_ds


