from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import LoadModel

prompt = ChatPromptTemplate.from_template("Scrivi una poesia sul {argomento}.")
output_parsers = StrOutputParser()
chain = prompt | LoadModel.model | output_parsers
answer = chain.invoke({"argomento": "le teste di cazzo"})
print(answer)