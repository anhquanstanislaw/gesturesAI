import sys
from train_model import GestureModel
import record

def main():
    print("\n1. Train Model (learn from clenched_fist.jsonl)")
    print("2. To record and recognize fist")
    print("3. to exit")

    
    while True:
        choice = input("\nEnter your choice (1-2): ").strip()
        if choice == '1':

            print("give model name, if not it is trained at defaulted: ")
            path_to_model = input().strip()
            if not path_to_model:
                path_to_model = "model_defaulted"
            gm = GestureModel(path_to_model)
            gm.run_full_pipeline()


        
        elif choice == '2':
            print("input model name file")
            path_to_model = input().strip()
            if not path_to_model:
                path_to_model = "model_defaulted"
            curr = record.Record(path_to_model)
            curr.run()
        elif choice == '3':
            print("\nGoodbye!")
            sys.exit(0)()
        else:
            print("\nPlease try again.")

if __name__ == "__main__":
    main()
