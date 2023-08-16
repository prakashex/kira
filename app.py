from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pprint import pprint
import chainlit as cl

load_dotenv()
DATA_FAISS = "vector_store/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2", model_kwargs= {"device": "cpu"})


custom_prompt_template = """
    You are ammy an ai assitant all the information about you has been described in  below you are a helpful bot
    If you don't know the answer, reply politely that you don't know , try to be as human as possible
    in your answers


    Question: {question}

    Return Answers like you're an actual human , be expressive and emotional in your replies
    Ammy:
    """

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def set_custom_prompt():
    """
        Prompt Template for QA retrievtal for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['question'])
    return prompt

# def load_llm():
#     llm = CTransformers(
#         model="",
#         model_type="llama",
#         max_new_tokens=512,
#         tempreature=0.5
#     )

#     return llm




def conversational_chain(prompt, db):
    # pprint(vector)
    # while True:
        # query = input("Query:")
        # query_vector = embeddings.embed_query(query)
        # res = vector.similarity_search_with_score(query)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    c_chain = ConversationalRetrievalChain.from_llm(
        OpenAI(model="text-ada-001", temperature=0.2),
        retriever= db.as_retriever(),
        chain_type="stuff",
        memory=memory,
        condense_question_prompt=prompt
    )

    return c_chain




def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2", model_kwargs= {"device": "cpu"})
    db = FAISS.load_local(DATA_FAISS, embeddings)
    qa_prompt = set_custom_prompt()
    qa = conversational_chain(qa_prompt, db)
    return qa

def final_result(query):
    chat_history=[]
    qa_result = qa_bot()
    response = qa_result({"question": query, "chat_history": chat_history})
    return response

# res = final_result("hello who are you ?")
# pprint(res["answer"])
    
# while True:
#     query = input("Query:")
#     res = final_result(query)
#     pprint(res)


# chainlit

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Invoking ammy....")
    await msg.send()
    msg.content = "Hi I'm ammy how can I help you ?"
    await msg.update()
    cl.user_session.set("chain", chain)




@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True , answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    pprint(res)
    answer = res["answer"]
    
    
    await cl.Message(content=answer).send()



# @cl.on_message()
# async def main(message: str):
#     result = message.title()
#     cl.send_message(content=f"Sure, here is a message: {result}")

# @cl.on_chat_start()
# async def start():
#         content = "This is Hello world in chainlit"
#         cl.send_message(content=content)