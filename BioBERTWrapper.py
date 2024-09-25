import torch
from transformers import BertTokenizer, BertForTokenClassification
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class BioBERTWrapper:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1"):
        print("Loading BioBERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_map = {
            1: 'Disease',
            2: 'Symptom',
            3: 'Medication',
            0: 'O'
        }

    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            is_split_into_words=False,
            max_length=512
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].cpu().numpy()[0])
        labels = predictions[0]
        return tokens, labels

    def extract_entities(self, tokens, labels):
        disease_entities = []
        symptom_entities = []
        medication_entities = []
        print("\nDetected Entities:")
        for token, label in zip(tokens, labels):
            clean_token = token.replace("##", "")
            entity_type = self.label_map.get(label, 'Unknown')
            if entity_type == 'Disease':
                disease_entities.append(clean_token)
            elif entity_type == 'Symptom':
                symptom_entities.append(clean_token)
            elif entity_type == 'Medication':
                medication_entities.append(clean_token)
        if disease_entities:
            print(f"Diseases: {', '.join(disease_entities)}")
        if symptom_entities:
            print(f"Symptoms: {', '.join(symptom_entities)}")
        if medication_entities:
            print(f"Medications: {', '.join(medication_entities)}")
        if not (disease_entities or symptom_entities or medication_entities):
            print("No entities detected.")

if __name__ == "__main__":
    report_text = "Raj, who had kidney cancer, liver cancer and sarcoma, now suffers from depression and fever."
    biobert = BioBERTWrapper()
    tokens, labels = biobert.predict(report_text)
    biobert.extract_entities(tokens, labels)
