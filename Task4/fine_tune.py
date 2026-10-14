import time
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

positive_texts = [
    "I love this product",
    "Excellent customer service",
    "Very happy with the purchase",
    "The quality is amazing",
    "Fast delivery and great packaging",
    "This product works perfectly",
    "Wonderful experience",
    "The support team was helpful",
    "Highly recommend this product",
    "Great value for money",
    "I am satisfied with the service",
    "Fantastic quality and performance",
    "The delivery was quick",
    "Easy to use and reliable",
    "The product exceeded expectations",
    "Friendly and professional support",
    "Very useful and convenient",
    "Good quality for the price",
    "I would buy this again",
    "The experience was excellent"
]

negative_texts = [
    "I hate this product",
    "Terrible customer service",
    "Very disappointed with the purchase",
    "The quality is awful",
    "Slow delivery and poor packaging",
    "This product does not work",
    "Horrible experience",
    "The support team was unhelpful",
    "I do not recommend this product",
    "Poor value for money",
    "I am unhappy with the service",
    "Terrible quality and performance",
    "The delivery was late",
    "Difficult to use and unreliable",
    "The product failed my expectations",
    "Rude and unprofessional support",
    "Very useless and inconvenient",
    "Bad quality for the price",
    "I would never buy this again",
    "The experience was disappointing"
]

texts = positive_texts * 10 + negative_texts * 10
labels = [1] * len(positive_texts * 10) + [0] * len(negative_texts * 10)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

train_dataset = Dataset.from_dict({
    "text": train_texts,
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "text": val_texts,
    "labels": val_labels
})

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels,
            predictions,
            average="macro"
        )
    }

training_args = TrainingArguments(
    output_dir="task4/results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

start_time = time.time()
trainer.train()
training_time = time.time() - start_time

metrics = trainer.evaluate()
print("\nFine-tuned DistilBERT Results")
print("Accuracy:", round(metrics["eval_accuracy"], 4))
print("Macro F1:", round(metrics["eval_f1"], 4))
print("Training time (seconds):", round(training_time, 2))

trainer.save_model("task4/saved_model")
tokenizer.save_pretrained("task4/saved_model")