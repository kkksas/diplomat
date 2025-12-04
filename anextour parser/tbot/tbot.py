from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from transformers import AutoModel
import time
import re, string
import pandas as pd
import numpy as np
import matplotlib

class Dummyatel():
  
  chroma_db: Chroma
  bm25: BM25Retriever
  documents: list
  
  def __init__(self):
    self.create_new_emb = False
    self.chunked = 'chunked_all.xlsx'
    self.tours_path = 'parsed.xlsx'
    self.persist_directory = 'chromadb_all'
    self.collection_name = 'v_russ'
    self.emb_model_name = "intfloat/multilingual-e5-large-instruct"
    self.k = 3
    self.threshold = 0.73
    self.tours = pd.read_excel(self.tours_path)
    self.tours['id'].apply(lambda x: int(x))
    self.tours.set_index(['id'])
    self.get_docs()
    model_kwargs = {'device': 'cuda'}
    self.emb = HuggingFaceEmbeddings(model_name=self.emb_model_name, model_kwargs=model_kwargs)#intfloat/multilingual-e5-large-instruct, ai-sage/Giga-Embeddings-instruct
    if self.create_new_emb:
      #создать эмбы
      self.chroma_db = Chroma.from_documents(self.documents, self.emb, persist_directory=self.persist_directory, collection_name=self.collection_name )
    else:
      #загрузить эмбы
      self.chroma_db = Chroma(persist_directory=self.persist_directory, collection_name=self.collection_name , embedding_function=self.emb)
    
    self.bm25_retriever = BM25Retriever.from_documents(
      documents=self.documents,
      preprocess_func=self.tokenize,
      k=self.k
    )
    
    self.ensemble_retriever = EnsembleRetriever(
    retrievers=[self.chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k':self.k, 'score_threshold': self.threshold}), 
                self.bm25_retriever],
      weights=[0.8, 0.2],
      tags=['e5', 'bm25']
    )
  
    
  def get_docs(self):
      cds = pd.read_excel(self.chunked, usecols=['id','context']).dropna()
      cds['len_c'] = cds['context'].apply(lambda x: len(x))
      
      documents = []
      for i in cds.iloc:
        documents.append(Document(page_content=i['context'], metadata = {"id":int(i['id'])}))
      self.documents = documents
    
    
  def sim_search_with_thresh(self, query):
      res = self.ensemble_retriever.invoke(query)
      ret_chunks=[]
      ids = []
      for elem in res:
          ret_chunks.append(elem.page_content)
          ids.append(elem.metadata['id'])
      # ret_chunks = list(elem[0].metadata['id'] for elem in res)
      return [query, ret_chunks, list(set(ids))]
    
  def tokenize(self, s: str) -> list[str]:
      """Очень простая функция разбития предложения на слова"""
      return s.lower().translate(str.maketrans("", "", string.punctuation)).split(" ")
    
  def get_recs_message(self, ids):
    msg = 'Вот несколько туров подходящих по запросу (наверное):\n\n'
    for id in ids:
      msg += f"Название тура: {self.tours['Name'].iloc[id]}\n"
      msg += f"Краткое описание: {self.tours['Short'].iloc[id]}\n\n"
        
    return msg
        
   



import telebot

API_TOKEN = '8483143274:AAEWtG-N2beGelk4FVqMYkxVRqhhC6AFC5Q'

bot = telebot.TeleBot(API_TOKEN)

retriever = Dummyatel()
print('я готоу')
# retriever.sim_search_with_thresh("Хочу тур по Санкт-Петербургу, включающий в себя экскурсии в Петергоф (Нижний парк) и Кронштадт")
# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Если ты не я, то пшел нахуй отсюда
""")

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    [query, ret_chunks, ids] = retriever.sim_search_with_thresh(message.text)
    msg = retriever.get_recs_message(ids)
      
    bot.reply_to(message, text = msg)


bot.infinity_polling()