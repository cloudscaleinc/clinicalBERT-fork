from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the conll2003 dataset with trust_remote_code=True
dataset = load_dataset("conll2003", trust_remote_code=True)

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

example_text = " ".join(dataset['train'][0]['tokens']) 
print(f"Input text: {example_text}")

# Run NER on the example text
ner_results = ner_pipeline(example_text)

# Output the NER results
print("NER Results:", ner_results)
