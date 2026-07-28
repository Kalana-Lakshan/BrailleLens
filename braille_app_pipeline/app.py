import os
import sys
import argparse
from model import predict_image
from audio_engine import speak, listen_spoken_answer

def main():
    parser = argparse.ArgumentParser(description="BrailleLens Interactive App")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.path.join(os.path.dirname(__file__), "braille_model.pth")

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train the model first by running train.py")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)

    print(f"Predicting character for image: {args.image}")
    try:
        predicted_char, confidence = predict_image(args.image, args.model)
    except Exception as e:
        print(f"Error predicting image: {e}")
        sys.exit(1)

    print(f"\n--- BrailleLens Pipeline ---")
    print(f"Prediction: '{predicted_char}' (Confidence: {confidence:.2f})")
    
    print("\nSelect Mode:")
    print("[1] Learning Mode")
    print("[2] Testing Mode")
    
    mode = input("Enter choice (1 or 2): ").strip()
    
    if mode == '1':
        print("\n--- LEARNING MODE ---")
        speak(f"This is character {predicted_char}")
    elif mode == '2':
        print("\n--- TESTING MODE ---")
        speak("Please state the character under your finger.")
        spoken = listen_spoken_answer(timeout=5)
        
        if spoken is None:
            # Re-prompt using keyboard fallback if STT totally failed and didn't trigger fallback
            spoken = input("Could not get answer. Please type it: ").strip().lower()
            
        if predicted_char in spoken:
            speak(f"Correct! You identified character {predicted_char} properly.")
        else:
            speak(f"Incorrect. You said {spoken}, but the correct character is {predicted_char}.")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
