from zhipuai import ZhipuAI

Access_Token = "d3cc78289acf4de296fba2b6d56a52eb"  # Competition team Token, used to access the competition database
MODEL = "glm-4-plus"  # Default large model used; this solution uses the GLM-4-PLUS model entirely


class LLMClient:
    def __init__(self):
        client = ZhipuAI(api_key="d25e7ccdd599490bb3113ad6ac8e515f.G0cQzHFMidedbXF5")
        self.client = client

    def generate(self, messages, model=MODEL):
        """
        Create a chat completion using the provided messages and model.

        Parameters:
              messages (list): A list of message dictionaries to pass to the model.
              model (str): The model name to use.

        Returns:
              response (dict): The response from the chat completion endpoint.
        """
        response = self.client.chat.completions.create(
            model=model, stream=False, messages=messages
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    client = LLMClient()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Please generate a random number between 1 and 10.",
        },
    ]
    response = client.generate(messages)
    print(response)
