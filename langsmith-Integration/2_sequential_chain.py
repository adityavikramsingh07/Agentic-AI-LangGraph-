from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    'run_name': 'sequential_chain_run',
    'tags': ['sequential_chain', 'google_genai'],
    'metadata': {'model': 'gemini-2.5-flash', 'temperature': 0.7, 'parser': 'StrOutputParser'}
}

result = chain.invoke({'topic': 'Employment in India'}, config=config)

print(result)
