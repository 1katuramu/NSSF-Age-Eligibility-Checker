import json
from nssf_api_client import get_health, get_model_info, retrain_model, predict_single_claim, predict_batch_claims

def interactive_cli():
    print("Welcome to the NSSF Eligibility API Client CLI!")
    print("Commands: health, model_info, predict, predict_batch, retrain, exit")

    while True:
        cmd = input("\nEnter command: ").strip().lower()

        if cmd == 'health':
            print(json.dumps(get_health(), indent=2))

        elif cmd == 'model_info':
            print(json.dumps(get_model_info(), indent=2))

        elif cmd == 'predict':
            print("Enter claim data as JSON (single line):")
            try:
                data = json.loads(input())
                response = predict_single_claim(data)
                print(json.dumps(response, indent=2))
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == 'predict_batch':
            print("Enter claims as JSON array (single line):")
            try:
                data = json.loads(input())
                response = predict_batch_claims(data)
                print(json.dumps(response, indent=2))
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == 'retrain':
            print("Retraining model...")
            response = retrain_model()
            print(json.dumps(response, indent=2))

        elif cmd == 'exit':
            print("Goodbye!")
            break

        else:
            print("Unknown command. Try again.")

if __name__ == "__main__":
    interactive_cli()
