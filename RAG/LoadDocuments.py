import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=(r"https://towardsdatascience.com/",
               r"https://www.ibm.com/blog/artificial-intelligence-trends/",
               r"https://www.kdnuggets.com/",
               r"https://deepai.org/"),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer()),
    encoding='utf-8',
)
print("info docs")
docs = loader.load()
print(docs)
len(docs)
docs[0].metadata
len(docs[1].page_content)
