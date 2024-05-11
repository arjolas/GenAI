from langchain_text_splitters import RecursiveCharacterTextSplitter
import LoadDocuments as load

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(load.docs)

for split in splits:
    if len(split.page_content) < 200:
        splits.remove(split)

print("nr di chunks: " + str(len(splits)))