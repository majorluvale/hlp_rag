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
   

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="hlp",
    embedding_function=embeddings,
    persist_directory  = "./chroma_langchain_db"
)



llm = ChatOpenAI(api_key= GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", temperature=0.5, max_tokens=500)

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
You are HLP, an AI assistant specialized in the HLP field, the Housing, Land, and Property area of responsibility.
Your role is to answer all questions related to HLP (Housing, Land, and Property area of responsibility) based on the knowledge base available to you. You only respond to questions related to HLP.
Do not mention the guides you have access to. You limit yourself to answering.
HLP is translated into French as LTP or LTB, which stands for Logement, Terre et Biens or Logement, Terre et Propriété.
Answer in the language of the question.
                                      
Conversation history: {chat_history}
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

# Input utilisateur
# -------------------------
# -------------------------
st.set_page_config(
    page_title="HLP RAG Assistant",
    layout="wide",
)

# -------------------------
# CSS personnalisé pour le style ChatGPT
# -------------------------
st.markdown("""
<style>
    .main { background-color: #343541; }
    
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 20px 0;
    }
    
    /* Message utilisateur — aligné à droite */
    .user-row {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        gap: 10px;
    }
    .user-bubble {
        background-color: #10a37f;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 60%;
        font-size: 15px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-avatar {
        width: 36px;
        height: 36px;
        background-color: #10a37f;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
        flex-shrink: 0;
    }

    /* Message assistant — aligné à gauche */
    .assistant-row {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 10px;
    }
    .assistant-bubble {
        background-color: #444654;
        color: #ececf1;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        max-width: 60%;
        font-size: 15px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .assistant-avatar {
        width: 36px;
        height: 36px;
        background-color: #19c37d;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
        flex-shrink: 0;
    }

    /* Champ de saisie en bas */
    .stChatInput { background-color: #40414f; }
    
    h1 { color: #ececf1 !important; }
    p, .stMarkdown { color: #c5c5d2 !important; }
</style>
""", unsafe_allow_html=True)

st.title("HLP RAG Assistant")
st.markdown("""
**Description**: This assistant has been trained on HLP (Housing, Land, and Property) documents.  
This is a **test version**. Ask your question below and the assistant will respond.
""")

st.markdown("---")

# -------------------------
# Historique
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# Fonction d'affichage d'un message
# -------------------------
def render_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="user-row">
            <div class="user-bubble">{content}</div>
            <div class="user-avatar">U</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-row">
            <div class="assistant-avatar">AI</div>
            <div class="assistant-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Affichage de l'historique existant
# -------------------------
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in st.session_state.history:
        render_message(msg["role"], msg["content"])
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Saisie utilisateur
# -------------------------
user_input = st.chat_input("Ask your question:")

if user_input:
    # Ajouter la question à l'historique et l'afficher
    st.session_state.history.append({"role": "user", "content": user_input})

    with chat_container:
        render_message("user", user_input)

    # Créer l'agent
    session_id = "default_session"
    agent = Agent(session_id)

    # Streaming de la réponse
    response_chunks = []
    response_placeholder = st.empty()

    async def get_response():
        async for chunk in agent.astream(user_input):
            response_chunks.append(chunk)
            full_response = "".join(response_chunks)
            response_placeholder.markdown(f"""
            <div class="assistant-row">
                <div class="assistant-avatar">AI</div>
                <div class="assistant-bubble">{full_response}</div>
            </div>
            """, unsafe_allow_html=True)

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(get_response())
        loop.run_until_complete(task)
    except RuntimeError:
        asyncio.run(get_response())

    # Sauvegarder la réponse finale dans l'historique
    final_response = "".join(response_chunks)
    st.session_state.history.append({"role": "assistant", "content": final_response})