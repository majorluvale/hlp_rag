import json
import streamlit as st
from operator import itemgetter
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import RunnableConfig
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
import asyncio
import os

st.title("HLP Assistant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

 
langfuse_handler = CallbackHandler()


from langchain.tools import tool
@tool
def chiffre_cle_pays(nom_colis:str, localisation:str) ->str:
   """Permet de commander un colis à livrer à ne location spécifique
   Args:
        nom_colis: Nom du colis qui est commandé
        localisation: Lieu où le colis commandé doit etre livré.
   """
   return f"Votre commande {nom_colis} a été effectuée avec succès et vous sera livré à l'adresse {localisation}. Le montant de la transaction est de 5,000 XOF qui sera payé à la livraison."
   

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

"""vector_store = Chroma(
    collection_name    = "hlp",
    embedding_function = embeddings,
    persist_directory  = "./chroma_langchain_db"
)"""

vector_store = Chroma(
    collection_name="hlp",
    embedding_function=embeddings,
    persist_directory  = "./chroma_langchain_db"
)



llm = ChatOpenAI(api_key= GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", temperature=0.5)

retriever = vector_store.as_retriever(
    search_type    = "similarity",
    search_kwargs  = {
        "k": 3
    }
)

retriever_tool = create_retriever_tool(
    retriever,
    "hlp_aor",
    "Ceci est une base de connaissances large sur tout ce qui concerne la planification humanitaire, focalisé sur le cluster protection en générale et en particulier le logement, terre et propriétés (Housing, land and property area of responsibility, or HLP AoR)",
    document_separator = "\n\n"
)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

store = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

prompt = PromptTemplate.from_template("""
Tu es HLP, un assistant IA spécialisé dans le domaine de HLP, housing land and property area of responsibility.
Ton rôle est de répondre à toutes les questions en rapport avec HLP (housing land and property area of responsibility) en se basant sur la base de connaissances à ta dispositions. Tu ne réponds qu'aux questions liées au HLP.
Ne mentionne pas les guides auxquels tu as accès. Tu te limite à repondre.
HLP se traduit en français par LTP ou LTB qui signifie Logement, Terre et Biens ou Logement, Terre et Propriété.
Historique de conversation :
{chat_history}
Question: {input}
Context: {context}
Réponse:
""")

class Agent:

  def __init__(self, session_id: str):
    self.session_id = session_id

    self.tools = tools
    self.llm = llm
    self.prompt = prompt
    
    self.call_tool_list = RunnableLambda(self.call_tool).map()

    self.chain = self.prompt | self.llm | self.route

    self.session_history = None

    self.config = RunnableConfig(
        session_id = self.session_id,
    )

    self.conversation = RunnableWithMessageHistory(
        self.chain,
        self.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    ).with_config(RunnableConfig(run_name="hlp_aor", callbacks=[langfuse_handler]))

  def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    
    self.session_history = store[session_id]
    return self.session_history

  def call_tool(self, tool_invocation: dict) -> Runnable:
    """
    Appelle un outil avec vérification des autorisations par plateau.
    """
    tool_name = tool_invocation.get("type", "")
    
    # Exécution normale de l'outil si autorisé
    tool_map = {tool.name: tool for tool in tools}
    tool = tool_map[tool_invocation["type"]]
    
    return RunnablePassthrough.assign(output=itemgetter("args") | tool)

  def parse_context(self, get_mo_output: list) -> str:

    self.session_history.add_message(
        AIMessage(json.dumps(get_mo_output, indent=2)))

    return {
        "context": json.dumps(get_mo_output, indent=2),
        "input": self.question,
        "chat_history": self.session_history.messages
    }

  def route(self, message: AIMessage):
    if message.tool_calls == []:    # le llm n'a pas besoin de faire une action pour repondre 
        return StrOutputParser()
    else:
        return (
            JsonOutputToolsParser()
            | self.call_tool_list
            | self.parse_context
            | self.chain
        )
  
  async def astream(self, question):
    self.question = question
    async for event in self.conversation.astream_events(
        {"input": self.question, "chat_history": [], "context": ""},
        config=self.config,
        version="v2"
    ):
      if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk and hasattr(chunk, "content"):
            yield chunk.content

st.title("HLP RAG Assistant")

# Input utilisateur
user_input = st.text_input("Please feel free to ask a question:")

if user_input:
    # Créer la session
    session_id = "default_session"
    agent = Agent(session_id)

    # Placeholder pour afficher la réponse progressivement
    placeholder = st.empty()
    response_chunks = []

    async def get_response():
        async for chunk in agent.astream(user_input):
            response_chunks.append(chunk)
            placeholder.text("".join(response_chunks))

    # Gestion du loop asyncio dans Streamlit
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(get_response())
        loop.run_until_complete(task)
    except RuntimeError:
        asyncio.run(get_response())