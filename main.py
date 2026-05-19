import argparse
import sys
import os

from src.data.scraper import build_corpus
from src.training.train import train
from src.inference.generate import run_inference

def main():
    parser = argparse.ArgumentParser(description="STC Transformer Project")
    parser.add_argument('action', choices=['scrape', 'train', 'generate', 'all'], 
                        help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action in ['scrape', 'all']:
        print("=== PHASE 1: Data Engineering ===")
        categories = [
            "Mining", "Quarrying", "Clay", "Iron ores", "Lithics", "Woodworking", 
            "Forestry", "Charcoal", "Smelting", "Charcoal ovens", "Survival skills",
            "Swords", "Metallurgy", "Blacksmithing"
        ]
        build_corpus(categories, "stc_training_data.txt", max_pages_per_cat=60)
        
    if args.action in ['train', 'all']:
        print("\n=== PHASE 2-4: Tokenization & Training ===")
        train()
        
    if args.action in ['generate', 'all']:
        print("\n=== PHASE 5: Inference ===")
        run_inference()

if __name__ == "__main__":
    main()
