import datetime
import gradio as gr
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


import warnings
warnings.filterwarnings('ignore')

current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
# print(llm_name)


def chatWithNCAIR(question, history):
    load_dotenv()

    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    template = """If the user says any greetings and says their name like "hello I'm bishop", always reply by greeting the person and saying his name  
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always be nice at the end of the answer. Make sure you remember the last conversation, especially if the user told you their name 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=template,)

    # Run chain
    from langchain.chains import RetrievalQA
    # question = "Will interns go through the fabLab during the onboarding?"
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
    )

    # result = qa({"question": question})
    result = qa_chain({"query": question})
    # return result["answer"]
    return result["result"]


demo = gr.ChatInterface(fn=chatWithNCAIR,
                        chatbot=gr.Chatbot(height=300, min_width=40),
                        textbox=gr.Textbox(
                            placeholder="Ask me a question relating to NCAIR"),
                        title="Chat with NCAIR💬",
                        description="Ask NCAIR any question",
                        theme="soft",
                        cache_examples=True,
                        retry_btn=None,
                        undo_btn="Delete Previous",
                        clear_btn="Clear",)

demo.launch(inline=False)
