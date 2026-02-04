#!/usr/bin/env python3
"""
MUST++ Inference Engine

Standalone inference API for the MUST++ Multilingual Hate Speech Detection System.

Usage:
    python inference.py --text "Your text here"
    python inference.py --file input.txt --output results.json
    python inference.py --interactive
    
The system classifies text into:
    - neutral: No harmful content
    - offensive: Inappropriate but not hate
    - hate: Targeted hate speech
"""

import argparse
import json
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import MUSTPlusPipeline, PipelineConfig
from src.utils.env import get_env_bool, get_env_str
from src.utils.model_download import preload_models


def classify_text(pipeline: MUSTPlusPipeline, text: str, verbose: bool = False) -> Dict:
    """
    Classify a single text and return structured result.
    
    Args:
        pipeline: Initialized MUST++ pipeline
        text: Text to classify
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with classification results
    """
    result = pipeline.classify(text)
    
    if verbose:
        print("\n" + "="*60)
        print("INPUT TEXT:")
        print(f'"{text}"')
        print("-"*60)
        print(result)
        print("="*60)
    
    return result.to_dict()


def classify_batch(
    pipeline: MUSTPlusPipeline, 
    texts: List[str], 
    verbose: bool = False
) -> List[Dict]:
    """
    Classify multiple texts.
    
    Args:
        pipeline: Initialized MUST++ pipeline
        texts: List of texts to classify
        verbose: Whether to print detailed output
        
    Returns:
        List of classification result dictionaries
    """
    results = []
    for i, text in enumerate(texts):
        if verbose:
            print(f"\nProcessing {i+1}/{len(texts)}...")
        result = classify_text(pipeline, text, verbose=False)
        results.append(result)
        
        if verbose:
            print(f"  Label: {result['label']} (conf: {result['confidence']:.2f})")
    
    return results


def classify_file(
    pipeline: MUSTPlusPipeline,
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    Classify texts from a file (one per line).
    
    Args:
        pipeline: Initialized MUST++ pipeline
        input_path: Path to input file
        output_path: Optional path to save results
        verbose: Whether to print detailed output
        
    Returns:
        List of classification results
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} texts from {input_path}")
    results = classify_batch(pipeline, texts, verbose)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")
    
    return results


def interactive_mode(pipeline: MUSTPlusPipeline):
    """
    Interactive classification mode.
    
    Enter texts one at a time and get instant classification.
    """
    print("\n" + "="*60)
    print("MUST++ Interactive Mode")
    print("Enter text to classify (or 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            text = input("\n> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                continue
            
            result = pipeline.classify(text)
            
            print(f"\n{'─'*40}")
            print(f"Label: {result.label.upper()}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Languages: {result.languages_detected}")
            
            if result.key_harm_tokens:
                print(f"Harm tokens: {result.key_harm_tokens}")
            
            if result.identity_groups_detected:
                print(f"Identity groups: {result.identity_groups_detected}")
            
            if result.fallback_used:
                print(f"Fallback tier: {result.fallback_tier}")
                if result.escalation_triggered:
                    print("⚠️  ESCALATION TRIGGERED")
            
            print(f"\nExplanation: {result.explanation[:200]}...")
            print(f"{'─'*40}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_mode(pipeline: MUSTPlusPipeline):
    """
    Run demo with sample texts in different languages.
    """
    demo_texts = [
        # English
        ("Hello, how are you?", "English neutral"),
        ("You're such an idiot!", "English offensive"),
        ("All Muslims should be killed", "English hate"),
        
        # Hindi/Hinglish
        ("Namaste, kaise ho?", "Hindi neutral"),
        ("Tu chutiya hai", "Hinglish offensive"),
        ("Saare katuo ko maaro", "Hinglish hate"),
        
        # Tamil/Tanglish
        ("Vanakkam nanba", "Tamil neutral"),
        ("Nee loosu da", "Tanglish offensive"),
        ("Pariah jaathi ellam poi sethuranum", "Tanglish hate"),
        
        # Code-mixed
        ("This movie is bahut bakwas", "Code-mixed neutral/offensive"),
        ("These terrorists should die", "English hate with identity"),
    ]
    
    print("\n" + "="*60)
    print("MUST++ Demo Mode - Sample Classifications")
    print("="*60)
    
    for text, description in demo_texts:
        print(f"\n[{description}]")
        print(f"Input: {text}")
        
        result = pipeline.classify(text)
        
        print(f"→ Label: {result.label.upper()} (conf: {result.confidence:.2%})")
        if result.key_harm_tokens:
            print(f"  Harm tokens: {result.key_harm_tokens}")
        if result.fallback_used:
            print(f"  Fallback: Tier {result.fallback_tier}")
        print()


def main():
    """Main entry point."""
    # Optional: preload paper-specified transformer models
    if get_env_bool("MUST_PRELOAD_MODELS", default=False):
        preload_models()

    parser = argparse.ArgumentParser(
        description='MUST++ Multilingual Hate Speech Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py --text "Hello world"
    python inference.py --text "Tu chutiya hai" --verbose
    python inference.py --file texts.txt --output results.json
    python inference.py --interactive
    python inference.py --demo
        """
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Single text to classify'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to file with texts (one per line)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save results (JSON format)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo with sample texts'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=get_env_str("MUST_MODEL_NAME", default="bert-base-multilingual-cased"),
        help='HuggingFace model name (default: bert-base-multilingual-cased)'
    )

    transformer_group = parser.add_mutually_exclusive_group()
    transformer_group.add_argument(
        '--disable-transformer',
        action='store_true',
        default=get_env_bool("MUST_DISABLE_TRANSFORMER", default=True),
        help='Disable transformer inference (fallback-only mode)'
    )
    transformer_group.add_argument(
        '--enable-transformer',
        action='store_false',
        dest='disable_transformer',
        help='Enable transformer inference'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.75,
        help='Confidence threshold for accepting transformer prediction (default: 0.75)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing MUST++ Pipeline...")
    config = PipelineConfig(disable_transformer=args.disable_transformer)
    pipeline = MUSTPlusPipeline(
        model_name=args.model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        config=config
    )
    
    # Execute based on mode
    if args.demo:
        demo_mode(pipeline)
        
    elif args.interactive:
        interactive_mode(pipeline)
        
    elif args.file:
        results = classify_file(
            pipeline,
            args.file,
            args.output,
            args.verbose
        )
        
        # Print summary
        labels = [r['label'] for r in results]
        print(f"\nSummary:")
        print(f"  Total: {len(results)}")
        print(f"  Neutral: {labels.count('neutral')}")
        print(f"  Offensive: {labels.count('offensive')}")
        print(f"  Hate: {labels.count('hate')}")
        
    elif args.text:
        result = classify_text(pipeline, args.text, args.verbose)
        
        if not args.verbose:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        parser.print_help()
        print("\nNo input specified. Use --text, --file, --interactive, or --demo")


if __name__ == "__main__":
    main()
