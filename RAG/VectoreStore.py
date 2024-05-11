import os
from langchain_community.vectorstores import FAISS
import SelectEmbedModel as embedd
import SplittingTests as doc

if os.path.exists("../ultimi_articoli"):
    faiss_index=FAISS.load_local("../ultimi_articoli", embeddings=embedd.embeddings,
                                 allow_dangerous_deserialization=True)
else:
    faiss_index=FAISS.from_documents(doc.splits, embedd.embeddings)
    faiss_index.save_local("ultimi_articoli")