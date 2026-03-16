import sys
from train_model import GestureModel

def main():
    print("\n1. Train Model (learn from clenched_fist.jsonl)")
    print("2. Exit")

    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '1':

        print("Starting model training...")
        gm = GestureModel()
        gm.run_full_pipeline()
        

    
    elif choice == '2':
        print("\nGoodbye!")
        sys.exit(0)
    
    else:
        print("\nPlease try again.")

if __name__ == "__main__":
    main()
