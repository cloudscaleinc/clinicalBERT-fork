import torch
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from preprocess import load_and_preprocess_data
import json

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Load dataset and tokenizer
    dataset, num_labels = load_and_preprocess_data()

    # Load the pre-trained ClinicalBERT model and move it to the appropriate device (GPU if available)
    model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_labels)
    model.to(device)  # Move the model to GPU or CPU

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Define training arguments, utilizing mixed precision and GPU-optimized settings
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Change to "epoch" for periodic evaluation
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],  # Adjust based on your GPU memory
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        gradient_accumulation_steps=8,  # Helps reduce memory load with smaller batches
        fp16=True,  # Mixed precision training for faster execution on GPU
        logging_dir='./logs',  # Logs for TensorBoard
        logging_steps=100,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_clinicalbert_ner")
    tokenizer.save_pretrained("./fine_tuned_clinicalbert_ner")

if __name__ == "__main__":
    main()
