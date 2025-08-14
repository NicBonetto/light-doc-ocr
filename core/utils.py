import evaluate
from core.model import processor

cer_metric = evaluate.load('cer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens = True)
    label_ids[ label_ids == -100 ] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens = True)
    cer = cer_metric.compute(predictions = pred_str, references = label_str)

    return { 'cer': cer }

