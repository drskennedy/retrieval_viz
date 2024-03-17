# main.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
import LoadVectorize
import timeit
from renumics import spotlight
import pandas as pd
import numpy as np

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def visualize_distance(db,question,answer,docs) -> None:
    embeddings_model = HuggingFaceEmbeddings()
    # FAISS data structures
    vs = db.__dict__.get("docstore")
    docstore_id_list = db.__dict__.get("index_to_docstore_id").values()
    doc_cnt = db.index.ntotal
    # create embeddings
    embeddings_vec = db.index.reconstruct_n()
    # create a list of lists
    doc_list = list()
    for i,id in enumerate(docstore_id_list):
        a_doc = vs.search(id)
        doc_list.append([a_doc.page_content,embeddings_vec[i],False])
    # create a list of lists for relevant docs
    for rel_doc in docs:
        doc_list.append([rel_doc.page_content,embeddings_model.embed_query(rel_doc.page_content),True])

    df = pd.DataFrame(doc_list,columns=['document','embedding','is_relevant'])

    # add rows for question and answer
    question_df = pd.DataFrame({
            "question": question,
            "embedding": [embeddings_model.embed_query(question)],
            "is_relevant": True,
        })
    answer_df = pd.DataFrame({
            "answer": answer,
            "embedding": [embeddings_model.embed_query(answer)],
            "is_relevant": True,
        })
    df = pd.concat([question_df, answer_df, df])
    spotlight.show(df)                                                                                                                    

# Prompt template 
qa_template = """<|system|>
You are a friendly chatbot who always responds in a precise manner. If answer is 
unknown to you, you will politely say so.
Use the following context to answer the question below:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""

# Create a prompt instance 
QA_PROMPT = PromptTemplate.from_template(qa_template)

llm = LlamaCpp(
    model_path="./models/tinyllama/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
    temperature=0.01,
    max_tokens=2000,
    top_p=1,
    verbose=False,
    n_ctx=2048
)

# create retrievers
db,bm25_r = LoadVectorize.load_db()
faiss_retriever = db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 4}, max_tokens_limit=2000)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_r,faiss_retriever],weights=[0.3,0.7])

query = 'What is a major issue with policy-based routing virtual in-path SteelHead deployment?'
#query = 'What is the purpose of a peering rule on SteelHead?'

# Custom QA Chain 
output_parser = StrOutputParser()
chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | QA_PROMPT
    | llm
    | output_parser
    )

result = chain.invoke(query)
print(f'---------------------\nQ: {query}\nA: {result}')

docs = ensemble_retriever.get_relevant_documents(query)
# visualize
visualize_distance(db,query,result,docs)
