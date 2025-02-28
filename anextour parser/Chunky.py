import pandas as pd

df = pd.read_excel('./parsed.xlsx')
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    keep_separator='end',
    is_separator_regex=False,
    separators=['.']
)
print(df)
def chunky(x):
    texts = text_splitter.create_documents([x[4]])
    for text in texts:
        text.metadata['source_id'] = x['id']
        text.metadata['len'] = len(text.page_content)             
    return texts

def chunky_hard_coded_overlap(x):
    overlap = 50
    texts = text_splitter.create_documents([x[4]])
    next = ''
    for text in reversed(texts):
        if next != '':
            text.page_content = text.page_content + next.page_content[:overlap]
        next = text    
        text.metadata['source_id'] = x['id']
        text.metadata['len'] = len(text.page_content)             
    return texts

 
texts = df.apply(chunky_hard_coded_overlap, axis=1)
chunks = []
for i in texts:
    for j in i:
        chunks.append([*df.iloc[j.metadata['source_id']][:5], j.page_content])
df1 = pd.DataFrame(chunks)
df1.to_excel("chunked1.xlsx", index= False)