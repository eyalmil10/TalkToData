from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()


def call_gemini_api(prompt: str, model: str = "gemini-3-flash-preview") -> str:
    """
    Call the Gemini API with the given prompt.
    
    Args:
        prompt: The input text to send to the model
        model: The model to use (default: gemini-3-flash-preview)
        
    Returns:
        The text response from the model
    """
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text


# Example usage
if __name__ == "__main__":
    result = call_gemini_api("Explain how AI works in a few words")
    print(result)