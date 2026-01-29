from lime.lime_text import LimeTextExplainer
import numpy as np
import torch
import torch.nn.functional as F

class HateSpeechExplainer:
    """
    Implements Section 7: Explainable AI Module
    """
    
    def __init__(self, class_names):
        """
        class_names: list of labels (e.g. ['Neutral', 'Offensive', 'Hate'])
        """
        self.explainer = LimeTextExplainer(class_names=class_names)

    def explain_traditional(self, pipeline, text_instance, num_features=10):
        """
        Explain a traditional sklearn pipeline prediction
        """
        # pipeline.predict_proba is what LIME needs
        exp = self.explainer.explain_instance(
            text_instance, 
            pipeline.predict_proba, 
            num_features=num_features
        )
        return exp

    def explain_transformer(self, model, tokenizer, text_instance, num_features=10):
        """
        Explain a HF Transformer prediction
        """
        
        # Define a wrapper function that LIME can call
        def predictor(texts):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            model.eval()
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()
            return probs

        exp = self.explainer.explain_instance(
            text_instance, 
            predictor, 
            num_features=num_features
        )
        return exp
