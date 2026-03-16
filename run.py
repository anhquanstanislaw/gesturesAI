#!/usr/bin/env python3
"""
Simple runner script for the gesture recognition system.
Provides menu to train model or run live recognition.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train_model import GestureModel
from recognize import GestureRecognizer

def main():
    print("\n" + "="*50)
    print("  GESTURE RECOGNITION SYSTEM")
    print("="*50)
    print("\n1. Train Model (learn from clenched_fist.jsonl)")
    print("2. Run Live Recognition (requires trained model)")
    print("3. Exit")
    print("\n" + "="*50)
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        print("\n" + "-"*50)
        print("Starting model training...")
        print("-"*50 + "\n")
        
        gm = GestureModel()
        gm.run_full_pipeline()
        
        print("\n" + "-"*50)
        print("✓ Training complete! Model saved.")
        print("You can now run option 2 for live recognition.")
        print("-"*50 + "\n")
    
    elif choice == '2':
        model_file = Path('gesture_model.h5')
        if not model_file.exists():
            print("\n✗ Error: Trained model not found!")
            print("Please train the model first (option 1)")
            return
        
        print("\n" + "-"*50)
        print("Starting live gesture recognition...")
        print("Press 'q' to quit")
        print("-"*50 + "\n")
        
        gr = GestureRecognizer()
        gr.run()
        gr.cleanup()
    
    elif choice == '3':
        print("\nGoodbye!")
        sys.exit(0)
    
    else:
        print("\n✗ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
