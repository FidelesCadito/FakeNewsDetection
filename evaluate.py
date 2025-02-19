from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer
from sklearn.metrics import classification_report
from preprocess import load_data, preprocess_text
import torch

def evaluate_model(data_path, model_path, model_type="bert"):
    """Evaluate trained BERT or RoBERTa model."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if model_type == "bert" else RobertaTokenizer.from_pretrained('roberta-base')
    model = BertForSequenceClassification.from_pretrained(model_path) if model_type == "bert" else RobertaForSequenceClassification.from_pretrained(model_path)

    df = load_data(data_path)
    df['label'] = df['label'].map({'real': 0, 'fake': 1})
    encodings = preprocess_text(df['text'], tokenizer)
    labels = torch.tensor(df['label'].values)

    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, axis=1)

    print(classification_report(labels.numpy(), predictions.numpy()))

if __name__ == "__main__":
    evaluate_model("../data/fake_news.csv", "../models/fake_news_bert", "bert")
    evaluate_model("../data/fake_news.csv", "../models/fake_news_roberta", "roberta")
