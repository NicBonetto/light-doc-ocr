from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from core.model import model, processor
from core.data import load
from core.utils import compute_metrics
from config import OUTPUT_DIR, BATCH_SIZE, EPOCHS

train_ds, eval_ds = load()

training_args = Seq2SeqTrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE,
    predict_with_generate = True,
    eval_strategy = 'epoch',
    logging_steps = 10,
    num_train_epochs = EPOCHS,
    save_total_limit = 1,
    fp16 = False
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    processing_class = processor.image_processor,
    data_collator = default_data_collator,
    compute_metrics = compute_metrics
)

if __name__ == '__main__':
    trainer.train()
