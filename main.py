from BioBERTWrapper import BioBERTWrapper

if __name__ == "__main__":
    biobert = BioBERTWrapper()
    report_text = input("Please enter the medical report text and press Enter when done:\n").strip()

    if not report_text:
        print("Error: No text entered.")
    else:
        tokens, labels = biobert.predict(report_text)
        biobert.extract_entities(tokens, labels)
