from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import Retriever as ret
import prompt
import LoadModel as model
def format_documents(documents):
    return "\n\n".join(doc.page_content for doc in documents)

rag_chain = (
        {
        "context": ret.retriever | format_documents,
        "question": RunnablePassthrough()
    }
        | prompt.prompt
        | model.model
        | StrOutputParser()
)