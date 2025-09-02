import os
import numpy as np
import evaluate
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

load_dotenv()

def main():
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    
    print("Loading IMDB dataset...")
    imdb = load_dataset("imdb")
    small_train_dataset = imdb["train"].shuffle(seed=42).select(range(3000))
    small_test_dataset = imdb["test"].shuffle(seed=42).select(range(300))

    print("Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
    tokenized_test = small_test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}

    repo_name = "distilbert-imdb-sentiment-analysis"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=repo_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
        hub_token=hf_token,
        hub_model_id=f"jhondoee/{repo_name}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Pushing model to Hugging Face Hub...")
    trainer.push_to_hub()
    print("Training and push complete!")

if __name__ == "__main__":
    main()
