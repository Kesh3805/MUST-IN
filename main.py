import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from src.utils.config import LABELS, LANGUAGES, SEED, LANGUAGE_BASE_MAP
from src.preprocessing.cleaner import TextPreprocessor
from src.models.language_id import LanguageIdentifier
from src.features.feature_extraction import FeatureExtractor
from src.models.classifiers import TraditionalClassifier, TransformerClassifier
from src.evaluation.metrics import Evaluator
from src.xai.explainer import HateSpeechExplainer
from src.utils.model_persistence import ModelManager, ResultsManager
from src.utils.results_summary import ResultsSummarizer
from src.utils.env import get_env_bool, get_env_str
from src.utils.model_download import preload_models
import torch

try:
    from src.features.bert_embeddings import BertEmbedder
except Exception:
    BertEmbedder = None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MUST-IN: Multilingual Hate Speech Detection Framework')
    parser.add_argument('--data', type=str, default='data/raw/sample_dataset.csv', 
                       help='Path to the input dataset')
    parser.add_argument('--save-models', action='store_true', 
                       help='Save trained models for later use')
    parser.add_argument('--run-dl', action='store_true', 
                       help='Run deep learning models (mBERT)')
    parser.add_argument('--run-xlm', action='store_true', 
                       help='Run deep learning model (XLM-RoBERTa)')
    parser.add_argument('--run-uncased', action='store_true', 
                       help='Run traditional models with uncased text')
    parser.add_argument('--run-bert-embed', action='store_true', 
                       help='Run traditional models on BERT embeddings')
    parser.add_argument('--generate-report', action='store_true', default=True,
                       help='Generate comprehensive results report')
    args = parser.parse_args()
    
    print("=== MUST-IN Framework Implementation ===")

    # Optional: preload paper-specified transformer models
    if get_env_bool("MUST_PRELOAD_MODELS", default=False):
        downloaded = preload_models()
        if downloaded:
            print(f"Preloaded models: {downloaded}")
        else:
            print("No models were preloaded. Check network or model names.")
    
    # Initialize managers
    model_manager = ModelManager()
    results_manager = ResultsManager()
    summarizer = ResultsSummarizer()
    
    # 1. Load Data
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Data not found at: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # Paper: duplicate removal
    before = len(df)
    df = df.drop_duplicates(subset=["text", "platform", "language", "label"]).reset_index(drop=True)
    after = len(df)
    if after != before:
        print(f"Removed duplicates: {before - after}")
    
    # Normalize labels to lowercase for consistent mapping
    df['label'] = df['label'].str.lower()
    
    # Label Encoding
    label_map = {label: idx for idx, label in enumerate(LABELS)}
    df['label_id'] = df['label'].map(label_map)

    # Base language mapping for LID (keep derivatives in dataset; LID predicts base)
    df['language_base'] = df['language'].map(lambda x: LANGUAGE_BASE_MAP.get(x, x))
    
    # 2. Preprocessing Pipeline
    print("\n--- Step 2: Preprocessing ---")
    preprocessor = TextPreprocessor()
    
    # Create versions of text
    # A) Preprocessed for Traditional (with normalization)
    df['clean_text'] = df.apply(lambda row: preprocessor.preprocess(row['text'], row['language']), axis=1)
    
    # B) Uncased for uncased models
    df['clean_text_uncased'] = df.apply(lambda row: preprocessor.preprocess(row['text'], row['language'], uncased=True), axis=1)
    
    print("Sample cleaned text:")
    print(df[['text', 'clean_text', 'label']].head())

    # Export cleaned dataset (UTF-8) for deliverables
    os.makedirs('data/processed', exist_ok=True)
    processed_path = 'data/processed/dataset_cleaned.csv'
    df.to_csv(processed_path, index=False, encoding='utf-8')
    print(f"Saved cleaned dataset to {processed_path}")
    
    # 3. Language Identification (Section 3)
    print("\n--- Step 3: Language Identification ---")
    # We use clean_text for features
    lid = LanguageIdentifier(method='tfidf')
    # For demo, we just train on the whole tiny dataset to show functionality
    # In real world, use a dedicated LID dataset
    lid.train(df['clean_text'], df['language_base'])
    
    pred_lang = lid.predict(["नमस्ते", "hello world", "nanba"])
    print(f"LID Predictions: {pred_lang}")
    
    # 4. Data Split (Section 1.5)
    # 80-20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label_id'], test_size=0.2, random_state=SEED, stratify=df['label_id']
    )

    # Optional: also keep uncased split for experiments (Section 2.4)
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        df['clean_text_uncased'], df['label_id'], test_size=0.2, random_state=SEED, stratify=df['label_id']
    )
    
    # 5. Traditional Models (Section 5.1)
    print("\n--- Step 5.1: Traditional Models ---")
    extractor = FeatureExtractor()
    evaluator = Evaluator(LABELS)

    vectorizers = {
        "Count": extractor.get_bow_vectorizer(),
        "TFIDF": extractor.get_tfidf_vectorizer(),
    }

    traditional_models = [
        ("MNB", "nb"),
        ("GNB", "gnb"),
        ("SVM", "svm"),
        ("RF", "rf"),
    ]

    trained_pipelines = {}
    model_metrics = {}
    
    for vec_name, vec in vectorizers.items():
        for model_name, model_type in traditional_models:
            run_name = f"{model_name}_{vec_name}"
            clf = TraditionalClassifier(model_type, vectorizer=vec)
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)

            y_prob = None
            if hasattr(clf.pipeline, "predict_proba"):
                try:
                    y_prob = clf.predict_proba(X_test)
                except Exception:
                    y_prob = None

            # Get metrics before evaluation (to capture them)
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            args.run_uncased
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': acc,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                    metrics['roc_auc'] = roc_auc
                except:
                    pass
            
            model_metrics[run_name] = metrics
            
            # Store results
            results_manager.add_experiment(run_name, metrics, {'vectorizer': vec_name, 'model_type': model_type})
            summarizer.add_result(run_name, metrics)
            
            # Run evaluation (prints results)
            evaluator.evaluate(y_test, y_pred, y_prob, run_name)
            trained_pipelines[run_name] = clf
            
            # Save model if requested
            if args.save_models:
                model_manager.save_model(clf.pipeline, run_name, metrics=metrics, 
                                        metadata={'vectorizer': vec_name, 'model_type': model_type})

    # Optional uncased traditional runs (helps compare cased vs uncased preprocessing)
    RUN_TRADITIONAL_UNCASED = False
    if RUN_TRADITIONAL_UNCASED:
        print("\n--- Step 5.1c: Traditional Models (Uncased Text) ---")
        for vec_name, vec in vectorizers.items():
            for model_name, model_type in traditional_models:
                run_name = f"{model_name}_{vec_name}_Uncased"
                clf = TraditionalClassifier(model_type, vectorizer=vec)
                clf.train(X_train_u, y_train_u)
                y_pred = clf.predict(X_test_u)

                y_prob = None
                if hasattr(clf.pipeline, "predict_proba"):
                    try:
                        y_prob = clf.predict_proba(X_test_u)
                    except Exception:
                        y_prob = None
                evaluator.evaluate(y_test_u, y_pred, y_prob, run_name)
                trained_pipelines[run_name] = clf

    # Optional: Traditional ML on BERT CLS embeddings (paper baseline)
    # This requires downloading a transformer model.
    RUN_BERT_EMBED_BASELINES = args.run_bert_embed
    if RUN_BERT_EMBED_BASELINES and BertEmbedder is not None:
        print("\n--- Step 5.1b: Traditional Models on BERT CLS Embeddings ---")
        embedder = BertEmbedder(model_name='bert-base-multilingual-cased')
        X_train_emb = embedder.encode(X_train.tolist())
        X_test_emb = embedder.encode(X_test.tolist())

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train_emb, y_train)
        y_pred_lr = lr.predict(X_test_emb)
        y_prob_lr = lr.predict_proba(X_test_emb)
        evaluator.evaluate(y_test, y_pred_lr, y_prob_lr, "LogReg_mBERTCLS")
    
    # 6. Deep Learning Models (Section 5.2)
    # Skipping heavy training in this script, providing the structure to run it.
    # To run DL: set RUN_DL = True
    RUN_DL = args.run_dl
    RUN_XLM = args.run_xlm
    
    if RUN_DL or RUN_XLM:
        print("\n--- Step 5.2: Deep Learning Models (Transformers) ---")
        # Note: HF Trainer needs numeric labels
        # Create a small validation split from training only (keep test held out)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )

        if RUN_DL:
            model_name = get_env_str("MUST_MODEL_NAME", default="bert-base-multilingual-cased")
            dl_classifier = TransformerClassifier(model_name, num_labels=3)
            dl_classifier.train(X_tr.tolist(), y_tr.tolist(), X_val.tolist(), y_val.tolist(), epochs=1)
            
            # Preds
            dl_preds = dl_classifier.predict(X_test.tolist())
            evaluator.evaluate(y_test, dl_preds, None, "mBERT_Cased")

        if RUN_XLM:
            xlm_classifier = TransformerClassifier('xlm-roberta-base', num_labels=3)
            xlm_classifier.train(X_tr.tolist(), y_tr.tolist(), X_val.tolist(), y_val.tolist(), epochs=1)
            
            # Preds
            xlm_preds = xlm_classifier.predict(X_test.tolist())
            evaluator.evaluate(y_test, xlm_preds, None, "XLM_RoBERTa")

        # Explain DL Model
        print("\n--- Step 7: XAI (Deep Learning) ---")
        explainer = HateSpeechExplainer(LABELS)
        idx = 0
        text_instance = X_test.iloc[idx]
        print(f"Explaining instance: '{text_instance}'")
        exp = explainer.explain_transformer(dl_classifier.model, dl_classifier.tokenizer, text_instance)
        # exp.save_to_file('results/lime_explanation_dl.html')

    # 7. XAI (Traditional)
    print("\n--- Step 7: XAI (Traditional) ---")
    explainer = HateSpeechExplainer(LABELS)
    
    # Explain on Hate/Offensive predictions (not only correct predictions)
    if len(X_test) > 0:
        # Prefer explaining an SVM model if available, but ensure we explain a Hate/Offensive prediction.
        preferred_order = [
            "SVM_TFIDF",
            "SVM_Count",
            "RF_TFIDF",
            "RF_Count",
            "MNB_TFIDF",
            "MNB_Count",
            "GNB_TFIDF",
            "GNB_Count",
        ]

        explain_key = None
        chosen = None
        chosen_preds = None
        chosen_idx = None

        keys = [k for k in preferred_order if k in trained_pipelines] + [k for k in trained_pipelines.keys() if k not in preferred_order]
        for k in keys:
            model = trained_pipelines[k]
            preds = model.predict(X_test)
            target_indices = [i for i, p in enumerate(preds) if int(p) in (1, 2)]
            if len(target_indices) > 0:
                explain_key = k
                chosen = model
                chosen_preds = preds
                chosen_idx = target_indices[0]
                break

        # Fallback: if no model predicts Offensive/Hate on this tiny sample split
        if chosen is None:
            explain_key = next(iter(trained_pipelines.keys()))
            chosen = trained_pipelines[explain_key]
            chosen_preds = chosen.predict(X_test)
            chosen_idx = 0

        preds = chosen_preds
        idx = chosen_idx

        text_instance = X_test.iloc[idx]
        true_label = LABELS[int(y_test.iloc[idx])]
        pred_label = LABELS[int(preds[idx])]
        print(f"Explaining instance: '{text_instance}' (Model: {explain_key}, Pred: {pred_label}, True: {true_label})")

        exp = explainer.explain_traditional(chosen.pipeline, text_instance)

        output_path = 'results/lime_explanation_traditional.html'
        exp.save_to_file(output_path)
        print(f"LIME explanation saved to {output_path}")
    
    # 8. Generate comprehensive results report
    if args.generate_report:
        print("\n--- Step 8: Generating Results Report ---")
        summarizer.generate_full_report()
        
        # Print summary
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Total models trained: {len(trained_pipelines)}")
        print(f"Results saved to: results/")
        if args.save_models:
            print(f"Models saved to: saved_models/")
        print("\nGenerated files:")
        print("  - results/results_summary.html (Open in browser for full report)")
        print("  - results/model_comparison.csv")
        print("  - results/model_comparison_plots.png")
        print("  - results/performance_heatmap.png")
        print("  - results/best_model_report.txt")
        print("  - results/lime_explanation_traditional.html")
        print("\nFor detailed analysis, run: jupyter notebook analysis.ipynb")
        print("="*70)

if __name__ == "__main__":
    main()
