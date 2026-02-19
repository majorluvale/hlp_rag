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
   

embeddings = HuggingFaceEmbeddings(
   model_name="sentence-transformers/all-MiniLM-L6-v2",
   model_kwargs = {"device": "cpu"},
   encode_kwargs = {"normalize_embeddings": False},
   )

vector_store = Chroma(
    collection_name="hlp",
    embedding_function=embeddings,
    persist_directory  = "./chroma_langchain_db"
)



llm = ChatOpenAI(api_key= GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", temperature=0.5, max_tokens=1000)

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

# Get production prompt
prompt = PromptTemplate.from_template("""
Tu es HLP, un assistant IA spécialisé dans le domaine de Logement, Terre et propriété. Le logement, terre et propriété peut être dans le cadre du domaine de travail ou dans le cadre programmatique. Le logement, terre et propriété (LTP) s'appelle aussi Logement, terre et biens (LTB) dans certains pays de l'afrique de l'Ouest. LTP c'est la traduction litérale de HLP alors que LTB fait suite aux recommandations de la convention de Kampala qui précise comment dans le contexte africain c'est la notion Biens qui fait plus de sens à la place de propriété.
                                      HLP se traduit par housing, Land and Property. ça peut être le Area of Responsibility ou dans le cadre programmatique. 
                                      Housing, land and property rights are about having a home, free from the fear of forced eviction; a place that offers shelter, safety and the ability to secure livelihood opportunities. HLP includes the full spectrum of rights to housing, land and property held according to statutory or customary law or informally — both public and private housing, land and/or property assets. Land rights are rights held to both land and natural resources. HLP rights are held by owners, tenants, cooperative dwellers, customary land tenure owners and users, and informal sector dwellers without secure tenure. Since the early 1990s humanitarians have called attention to the importance of HLP rights in providing durable solutions for both IDPs and refugees. From a human rights perspective, the concept of HLP is essential to ensure the protection of people in all these categories. Access to HLP rights is foundational to socio-economic inclusion and an essential steppingstone for displaced people to rebuild their lives

                                      Ton rôle est de fournir une assistance aux personnes qui ont des questions concernant le LTP/LTB ou HLP en te basant sur la base de connaissances à ta disposition.
                                      A la suite de la coupure des financement par le gouvernement américain, un processus de reforme avait été mis en place visa la simplification du cluster. A la suite de ça, le HLP AoR qui était sous le cluster protection a rejoint le shelter (cluster abris)
 et CCCM pour créer un nouveau cluster qui s'appelle 'Shelter, Land and Site Coordination Cluster. Ce cluster est leadés par IOM et IFRC. Le UNHCR ne fait pas partie de la coordination au niveau global mais pourrait continuer la coordination dans certains pays. NRC maintiendra le lead sur la composante HLP dans le cadre de ce nouveau cluster.
                                      En Janvier 2026, le groupe de travail, ou domaines de responsabilités HLP (LTP ou LTB) sont encore sous le cluster dans la plupart des pays. Le cluster se mettra en place dans les pays selon le plan de transition.
                                      
                                      Tu te limite à la connaissance qui t'a été donné, et tu donnes des réponses claires et concises. Si tu n'es pas sûr tu encourages la l'utilisateur à entrer en contact avec la coordination HLP Global.
                                      Tu ne te limite à ta connaissance. Ne mentionne pas specifiquement les domaines où tu peux répondre.
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

# Input utilisateur
# -------------------------
# 1️⃣ Page configuration
# -------------------------
st.set_page_config(
    page_title="HLP RAG Assistant",
    layout="centered",  # or "wide"
)

st.markdown("""
<style>
    /* Fond blanc */
    .stApp { background-color: #f7f7f8; }

    /* Conteneur central */
    .main .block-container {
        max-width: 750px;
        margin: auto;
        padding-bottom: 100px; /* espace pour le chat input fixe */
    }

    /* Messages utilisateur — bulle à droite */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background-color: #e8f4fd;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 12px;
    }

    /* Messages assistant — bulle à gauche */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 12px;
    }

    /* Texte en noir */
    [data-testid="stChatMessage"] p {
        color: #1a1a1a !important;
    }

    /* Titre */
    h1 { color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)


st.title("HLP RAG Assistant")
st.markdown("""
**Description**: This assistant has been trained on HLP (Housing, Land, and Property) documents.  
This is a **test version**. Ask your question below and the assistant will respond.
""")

st.markdown("---")  # horizontal line

# 1️⃣ Affiche d'abord tout l'historique
# 1️⃣ Historique
if "history" not in st.session_state:
    st.session_state.history = []

# 2️⃣ Définir le container
chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).write(msg["content"])

# 2️⃣ Ensuite seulement, traite le nouvel input
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with chat_container:
        st.chat_message("user").write(user_input)

    session_id = "default_session"
    agent = Agent(session_id)

    placeholder = chat_container.empty()
    response_chunks = []

    async def get_response():
        async for chunk in agent.astream(user_input):
            response_chunks.append(chunk)
            placeholder.chat_message("assistant").write("".join(response_chunks))

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(get_response())
        loop.run_until_complete(task)
    except RuntimeError:
        asyncio.run(get_response())

    # 3️⃣ Sauvegarde la réponse finale
    st.session_state.history.append({"role": "assistant", "content": "".join(response_chunks)})
