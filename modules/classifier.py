import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Classifier():
    def __init__(self, tags, model_path="models/classifier_model/"):
        self.tags = tags
        self.checkpoint_path = model_path  # Path to your fine-tuned model
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path, use_auth_token=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_path)
        self.articles = []
        self.not_articles = []
        
    def preprocess_inference_(self, text):
        inputs = self.tokenizer(text, truncation=True, return_tensors="pt")
        return inputs

    def generate_(self, text):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            inputs = self.preprocess_inference_(text)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = softmax(logits, dim=1)
            return probabilities
        
    def predict_(self, text):
        probabilities = self.generate_(text)
        predict_ed_class = torch.argmax(probabilities, dim=1).item()
        return predict_ed_class
    
    def classify(self):
        """Classifies the tags as articles or not and stores results in Instance's metadata
        """
        self.articles = []
        self.not_articles = []
        for tag in self.tags:
            prediction = self.predict_(tag)
            if prediction:
                self.articles.append(tag)
            else:
                self.not_articles.append(tag)
    
    def get_articles(self):
        """Returns the articles that were positively classified, Must run classify() first

        Returns:
            List[str]: List of html snippets that contain info about specific news articles
        """
        return self.articles
