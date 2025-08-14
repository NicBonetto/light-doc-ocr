from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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
    logging_steps = 50,
    num_train_epochs = EPOCHS,
    save_total_limit = 1,
    remove_unused_columns = False,
    learning_rate = 5e-5,
    fp16 = False
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = train_ds,
    eval_dataset = eval_ds,
    processing_class = processor.image_processor,
    compute_metrics = compute_metrics
)

def train_save():
    trainer.train()
    trainer.save_model('./model')
    processor.save_pretrained('./model')


if __name__ == '__main__':
    train_save()
