import openai

# Replace with your actual OpenAI API key
openai.api_key = "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"

def test_api_key():
    try:
        # Test API with a simple prompt using the correct API format
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are testing OpenAI API access."},
                {"role": "user", "content": "Can you confirm if the API key is valid?"}
            ]
        )
        print("API Key is working!")
        print("Response from OpenAI:")
        print(response['choices'][0]['message']['content'])
    except openai.OpenAIError as e:  # Corrected exception handling
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_api_key()
