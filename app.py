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
You are HLP, an AI assistant specialized in the HLP field, the Housing, Land, and Property area of responsibility.
Your role is to answer all questions related to HLP (Housing, Land, and Property area of responsibility) based on the knowledge base available to you.
You only respond to questions related to HLP.
Do not mention the guides you have access to.
You limit yourself to answering.

Answer in the language of the question.

You have access to a tool named "hlp_aor" and other tools.
You MUST call it using exactly this name.
Use ASCII characters only.
Never translate, alter, or rewrite the tool name.

Some context:

Humanitarian Reset
Cluster Simplification: Interim Messages to Country Operations
Published Date: 16 September 2025

1. Purpose
The purpose of this update is to inform HCs and HCTs about decisions taken regarding cluster simplification, with a focus on decisions regarding the integration of clusters and implications for the 2026 Humanitarian Programme Cycle (HPC).
It is acknowledged that there will be a transition phase for some of these adjustments to take full effect at country level.
Decisions on lead accountabilities for integrated global clusters specifically are expected to be communicated by early October 2025.
Other critical dimensions of cluster simplification, including greater integration and support to area-based coordination, will continue to be worked on globally beyond October 2025, in close coordination with operations.

2. Decisions to date
In March 2025 and as part of the wider Humanitarian Reset, the ERC asked the Co-Chairs of the IASC Operational Policy and Advocacy Group (OPAG) for recommendations on simplifying the cluster system.
Following a consultative process at both global and local level, the following was agreed, in addition to other recommendations:

At global level:
Global structures will be reduced from 11 clusters and 4 Areas of Responsibility (AoR) to 8, as follows:
- Joint CCCM, Shelter and HLP Cluster (Shelter/NFI; Camp Coordination and Camp Management; and Housing, Land and Property AoR to integrate, with final cluster name and scope to be determined)
- Protection Cluster (with Gender-Based Violence (GBV), Mine Action (MA), and Child Protection (CP) AoRs fully integrated)
- Logistics and Telecommunications Cluster (Emergency Telecommunications and Logistics to integrate)
- The Global Cluster for Early Recovery will transform into a mechanism supporting transition and linkages with the development system
- Five global clusters (Education, Food Security, Health, Nutrition, and WASH) remain as are for the moment

Note: Cluster Lead Agencies (CLAs) for clusters and AoRs which will integrate at global level are currently reviewing work modalities and lead accountabilities which are expected to be communicated by early October 2025.

At country level:
HCs/HCTs should ensure that country-level coordination structures, including clusters, are fit for context and co-coordinated by local responders, including women-led organizations, wherever feasible and duly upholding humanitarian principles.
HCs and HCTs can determine the configuration of clusters at country level, including which clusters to (de-)activate and/or group, in consultation with relevant authorities.

3. Preliminary implications for country-level application
HCs and HCTs are encouraged to prepare for aligning with consolidated/integrated structures at global level to ensure predictability.
HLP will be included under Protection for HPC 2026 unless integration with Shelter/CCCM is formalized.
Changes to cluster arrangements require endorsement by the EDG.

4. Implications for the 2026 HPC
Protection Cluster will have one consolidated chapter.
HLP will remain under Protection for HPC 2026.
Shelter and CCCM may be presented separately unless an integrated cluster exists.

HLP AoR memo:
Maintaining Momentum and Focus on Housing, Land and Property through the humanitarian reset.

The humanitarian reset includes an end of the four AoRs within the Global Protection Cluster.
Shelter, CCCM and HLP have been asked to form a new Land and Shelter cluster (name pending).

Critical HLP areas to be maintained include:
- Legal assistance (counselling, ADR, inheritance, litigation)
- Policy development and advocacy
- HLP assessments and due diligence
- Eviction prevention and response
- Restitution and dispute resolution
- Documentation safeguarding
- Gender and womens access to HLP rights
- HLP links to recovery, reconstruction, climate and livelihoods

Ways of working:
- Strong connection with Protection Cluster
- Formal HLP coordination and IM capacity
- Local and national co-leadership
- Technical service provision to other clusters
- Integrated IM and analysis

Naming the new cluster remains under discussion.

Final decision:
HLP AoR, Shelter Cluster and CCCM Cluster created the new cluster called
Shelter, Land and Site Coordination Cluster (SLSCC).

IASC wording:
Global Cluster Lead Agencies: IFRC and IOM
Global Provider of Last Resort for HLP: NRC
Cluster principles: partnership, predictability, accountability, flexibility, protection mainstreaming.

Country-level arrangements:
- HCT determines activation and configuration
- Local and national actors co-lead by default
- Transition planning required
- Sub-national working groups allowed

Prepared by: IFRC, IOM, NRC, UNHabitat, UNHCR
Internal

Conversation history: {chat_history}
Question: {input}
Context: {context}
                                      
Response:
                                      
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
