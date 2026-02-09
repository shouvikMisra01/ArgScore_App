import os
import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No API key found in .env")
    exit(1)

client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-flash-1.5",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-1.0-pro",
    "gemini-pro"
]

print(f"Testing {len(models_to_test)} models with API key: {api_key[:5]}...")

for model in models_to_test:
    print(f"\nTesting model: {model}")
    try:
        response = client.models.generate_content(
            model=model, 
            contents="Hello, are you working?"
        )
        print(f"✅ SUCCESS: {model}")
        print(f"Response: {response.text[:50]}...")
        break # Found a working one!
    except Exception as e:
        print(f"❌ FAILED: {model}")
        print(f"Error: {e}")
