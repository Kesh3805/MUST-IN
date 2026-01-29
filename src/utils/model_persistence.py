"""
Model Persistence Utilities
Provides functionality to save and load trained models
"""
import joblib
import os
import json
from datetime import datetime


class ModelManager:
    """
    Manages saving and loading of trained models and their metadata
    """
    
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model, model_name, metrics=None, metadata=None):
        """
        Save a trained model along with its metrics and metadata
        
        Args:
            model: The trained model/pipeline to save
            model_name: Name identifier for the model
            metrics: Dictionary of evaluation metrics
            metadata: Additional information about the model
        
        Returns:
            str: Path to the saved model
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = model_name.replace(' ', '_').replace('/', '_')
        model_path = os.path.join(self.models_dir, f"{safe_name}_{timestamp}.pkl")
        
        # Save the model
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metrics and metadata
        if metrics or metadata:
            info = {
                'model_name': model_name,
                'timestamp': timestamp,
                'metrics': metrics or {},
                'metadata': metadata or {}
            }
            info_path = model_path.replace('.pkl', '_info.json')
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
            print(f"Model info saved to: {info_path}")
        
        return model_path
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model file
        
        Returns:
            The loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Try to load associated info
        info_path = model_path.replace('.pkl', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"Model info loaded: {info['model_name']} (trained: {info['timestamp']})")
            return model, info
        
        return model, None
    
    def list_saved_models(self):
        """
        List all saved models in the models directory
        
        Returns:
            list: List of dictionaries containing model information
        """
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.pkl'):
                model_path = os.path.join(self.models_dir, file)
                info_path = model_path.replace('.pkl', '_info.json')
                
                model_info = {
                    'path': model_path,
                    'filename': file
                }
                
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    model_info.update(info)
                
                models.append(model_info)
        
        return sorted(models, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def get_best_model(self, metric='accuracy'):
        """
        Get the best performing model based on a specific metric
        
        Args:
            metric: The metric to use for comparison (default: 'accuracy')
        
        Returns:
            tuple: (model, info) for the best model, or (None, None) if no models found
        """
        models = self.list_saved_models()
        
        if not models:
            return None, None
        
        # Filter models that have the specified metric
        models_with_metric = [m for m in models if metric in m.get('metrics', {})]
        
        if not models_with_metric:
            print(f"No models found with metric: {metric}")
            return None, None
        
        # Find best model
        best_model_info = max(models_with_metric, key=lambda x: x['metrics'][metric])
        
        # Load the best model
        model, info = self.load_model(best_model_info['path'])
        
        return model, info


class ResultsManager:
    """
    Manages experiment results and tracks performance across multiple runs
    """
    
    def __init__(self, results_file='results/experiment_results.json'):
        self.results_file = results_file
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        self.results = self._load_results()
    
    def _load_results(self):
        """Load existing results from file"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {'experiments': []}
    
    def _save_results(self):
        """Save results to file"""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def add_experiment(self, model_name, metrics, config=None):
        """
        Add a new experiment result
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            config: Configuration used for the experiment
        """
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'config': config or {}
        }
        
        self.results['experiments'].append(experiment)
        self._save_results()
        print(f"Experiment result saved for: {model_name}")
    
    def get_experiments(self, model_name=None):
        """
        Get experiment results, optionally filtered by model name
        
        Args:
            model_name: Optional model name to filter by
        
        Returns:
            list: List of experiment results
        """
        experiments = self.results['experiments']
        
        if model_name:
            experiments = [e for e in experiments if e['model_name'] == model_name]
        
        return experiments
    
    def get_best_experiment(self, metric='accuracy'):
        """
        Get the best experiment based on a specific metric
        
        Args:
            metric: The metric to use for comparison
        
        Returns:
            dict: The best experiment result
        """
        experiments = [e for e in self.results['experiments'] 
                      if metric in e.get('metrics', {})]
        
        if not experiments:
            return None
        
        return max(experiments, key=lambda x: x['metrics'][metric])
    
    def compare_models(self, metric='accuracy'):
        """
        Compare all models based on a specific metric
        
        Args:
            metric: The metric to use for comparison
        
        Returns:
            list: Sorted list of (model_name, metric_value) tuples
        """
        model_scores = {}
        
        for exp in self.results['experiments']:
            if metric in exp.get('metrics', {}):
                model_name = exp['model_name']
                score = exp['metrics'][metric]
                
                # Keep best score for each model
                if model_name not in model_scores or score > model_scores[model_name]:
                    model_scores[model_name] = score
        
        return sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
