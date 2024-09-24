from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_and_align_labels(examples, tokenizer):
    # Tokenize the input tokens
    tokenized_inputs = tokenizer(
        examples['tokens'],
        padding='max_length',  # Ensure consistent padding
        truncation=True,
        max_length=512,
        is_split_into_words=True
    )

    # Align the labels (ner_tags) with the tokenized inputs
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to original words
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:  # Special tokens get -100
                label_ids.append(-100)
            elif word_id != previous_word_id:  # Only label the first token of a word
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)  # Padding for other sub-tokens
            previous_word_id = word_id

        # Pad labels to the same length as tokenized inputs
        label_ids += [-100] * (tokenized_inputs['input_ids'][i].count(0) - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels  # Add aligned labels to the tokenized inputs
    return tokenized_inputs


def load_and_preprocess_data():
    # Load the BC2GM corpus dataset
    dataset = load_dataset("spyysalo/bc2gm_corpus")

    # Load the tokenizer for Bio_ClinicalBERT
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Tokenize and align labels
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # Get the number of unique labels (for model initialization)
    num_labels = len(dataset['train'].features['ner_tags'].feature.names)

    return tokenized_datasets, num_labels
