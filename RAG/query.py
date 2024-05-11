import CreateChain as rag

def query(query):
    answer = rag.rag_chain.invoke(query)
    return answer

def query_stream(query):
    for chunk in rag.rag_chain.stream(query):
        print(chunk, end="", flush=True)