from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url = 'http://127.0.0.1:11434/v1',
    temperature = 0,
    api_key = 'not - needed',
    model_name = 'gemma'
)