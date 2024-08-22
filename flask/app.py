from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": 'Flask server is running'})

import queue
q = queue.Queue()
stop_item = "###finish###"

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
class StreamingStdOutCallbackHandlerYield(StreamingStdOutCallbackHandler):
    def on_llm_start(
        self, serialized, prompts, **kwargs
    ) -> None:
        """Run when LLM starts running."""
        with q.mutex:
            q.queue.clear()

    def on_llm_new_token(self, token, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
        q.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        """Run when LLM ends running."""
        q.put(stop_item)

from flask import request, Response, stream_with_context
from langchain_community.chat_models import ChatOpenAI
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
@app.route('/', methods=['POST'])
def send_prompt():
    data = request.get_json()
    prompt = data['prompt']
    
    llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo',
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandlerYield()]
    )

    DB_PATH = "vectorstores/db/"

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=GPT4AllEmbeddings())

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        condense_question_llm=llm,
        return_source_documents=True,
        chain_type="stuff",
        get_chat_history=lambda h : h
    )

    PROMPT_TEMPLATE = """You are a good assistant that answers questions. Your knowledge is strictly limited to the following piece of context. Use it to answer the question at the end.
    If the answer can't be found in the context, just say you don't know. *DO NOT* try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
    Give a response in the same language as the question.
    
    Context: {context}
    """
    system_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt

    ai_response = chain({"question": prompt, 'chat_history': ''})
    
    return Response(stream_with_context(ai_response['answer']))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
data = []
txt_loader = DirectoryLoader(
        "txtdata", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
data.extend(txt_loader.load())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[
    "\n\n",
    "\n",
    ]
)
texts=text_splitter.split_documents(data)
DB_PATH = "vectorstores/db/"
vectorstore = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(), persist_directory=DB_PATH)      
vectorstore.persist()        
if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5328)