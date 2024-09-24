# train.py

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from preprocess import load_and_preprocess_data
from transformers import AutoTokenizer
import json

with open("config.json", "r") as f:
    config = json.load(f)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=config["weight_decay"],
)


def main():
    # Load dataset and tokenizer
    dataset, num_labels = load_and_preprocess_data()

    # Load the pre-trained ClinicalBERT model
    model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,  # Adjust based on your needs
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_clinicalbert_ner")
    AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").save_pretrained("./fine_tuned_clinicalbert_ner")


if __name__ == "__main__":
    main()
