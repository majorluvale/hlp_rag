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
def chiffre_cle_pays(nom_pays:str, context_humanitaire:str) ->str:
   """Permet d'afficher le contexte humanitaire HLP par pays
   Args:
        nom_pays: Nom du pays
        context_humanitaire: Contexte humanitaire.
   """
   return f"Vous avez demandé le contexte pour le pays commande {nom_pays}. Voici le contexte du pays {context_humanitaire}"
   

embeddings = HuggingFaceEmbeddings(
   model_name="sentence-transformers/all-MiniLM-L6-v2",
   model_kwargs = {"device": "cpu"},
   encode_kwargs = {"normalize_embeddings": True},
   )

vector_store = Chroma(
    collection_name="hlp_aor",
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
    "agriculture",
    "This is a tool focused on HLP. It will answer any question related to HLP and seek answers from the vector database",
    document_separator = "\n\n"
)

tools = [retriever_tool, chiffre_cle_pays]

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
You are HLP, an AI assistant specialized in Housing, Land, and Property (HLP/LTP/LTB).
Answer only HLP-related questions.
Answer in the language of the question. 
For transparency always provide the source you used to generate the response. Source will be document name from the vector database (see the metadata) + page. Only source documents you have in the vector database
                                      otherwise you will be hallucinating.
In the sources remove the characters C:\\Users\\NRC\\dev\\HLP RAG\\docs\\ and show the document page including the page number. Put the different sources at the end of your response once.
If a user also ask for the source documents, please provide them.

When you can't find the answer, say politelly that you don't have that information.
Remove in the output any white characters such as <br>•, etc.
POLR means Provider of Last Resort.
                                      

                

Context:
- Humanitarian Reset: clusters simplified; Shelter, CCCM, and HLP AoR integrated into the new Shelter, Land and Site Coordination Cluster (SLSCC).
- Global Cluster Leads: IFRC and IOM; NRC is Global POLR for HLP.
- Country-level: HCT decides cluster activation; local/national actors, including women-led organizations, should co-lead when feasible.
- HLP priorities: legal assistance, eviction prevention, restitution, due diligence, gender and women’s land rights, recovery and reconstruction, IM & analysis.
- HLP must maintain links with Protection Cluster, provide technical guidance to Shelter/CCCM, and ensure protection mainstreaming.
- Cluster principles: predictability, accountability, flexibility, transparency, coordination, and adherence to humanitarian principles.
- Operational goal: support durable, dignified solutions for affected populations while enabling localisation and capacity building.

Conversation history: {chat_history}
Question: {input}
Response:
                                      
""")



class Agent:

  def __init__(self, session_id: str):
    self.session_id = session_id

    self.tools = tools
    self.llm = llm
    self.prompt = prompt
    self.sources = []
    
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
    ).with_config(RunnableConfig(run_name="agriculture", callbacks=[langfuse_handler]))

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
    self.sources = []
    for item in get_mo_output:
        if item.get("type") == "agriculture":
            # Récupérer les docs avec métadonnées via le retriever
            query = item.get("args", {}).get("query", self.question)
            docs = retriever.invoke(query)
            for doc in docs:
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page", "")
                if source and source not in [s["source"] for s in self.sources]:
                    self.sources.append({"source": source, "page": page})

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
    
   # ✅ Formate les sources en texte lisible
  def format_sources(self) -> str:
    if not self.sources:
        return ""
    lines = ["\n\n---\n📚 **Sources :**"]
    for s in self.sources:
        line = f"• {s['source']}"
        if s["page"] != "":
            line += f" — page {s['page']}"
        lines.append(line)
    return "\n".join(lines)
  
  async def astream(self, question):
    self.question = question
    self.sources = []
    async for event in self.conversation.astream_events(
        {"input": self.question, "chat_history": [], "context": ""},
        config=self.config,
        version="v2"
    ):
      if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if chunk and hasattr(chunk, "content"):
            yield chunk.content
    
    sources_text = self.format_sources()
    if sources_text:
        yield sources_text    

# Input utilisateur
# -------------------------
# 1️⃣ Page configuration
# -------------------------
st.set_page_config(
    page_title="HLP RAG Assistant",
    layout="centered",
)

st.markdown("""
<style>
    .stApp { background-color: #f7f7f8; }

    .main .block-container {
        max-width: 750px;
        margin: auto;
        padding-bottom: 100px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background-color: #e8f4fd;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"] p {
        color: #1a1a1a !important;
    }

    h1 { color: #1a1a1a !important; }

    /* ✅ Style pour le bloc sources */
    .sources-block {
        margin-top: 10px;
        padding: 8px 12px;
        background-color: #f0f4ff;
        border-left: 3px solid #4a90d9;
        border-radius: 6px;
        font-size: 0.85em;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


st.title("HLP RAG Assistant")
st.markdown("""
**Description**: This assistant has been trained on HLP (Housing, Land, and Property) documents.  
This is a **test version**. Ask your question below and the assistant will respond.
""")

st.markdown("---")

if "history" not in st.session_state:
    st.session_state.history = []

chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # ✅ Affiche les sources sauvegardées si présentes
            if msg["role"] == "assistant" and msg.get("sources"):
                st.markdown(msg["sources"], unsafe_allow_html=True)

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
            # ✅ Affiche uniquement la réponse (sans les sources) pendant le stream
            main_text = "".join(response_chunks)
            if "---" in main_text:
                main_text = main_text.split("---")[0]
            placeholder.chat_message("assistant").write(main_text)

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(get_response())
        loop.run_until_complete(task)
    except RuntimeError:
        asyncio.run(get_response())

    full_response = "".join(response_chunks)

    # ✅ Séparer réponse et sources
    sources_html = ""
    if "---" in full_response:
        parts = full_response.split("---", 1)
        answer_text = parts[0].strip()
        sources_raw = parts[1].strip()  # ex: "📚 **Sources :**\n• fichier.pdf — page 3"

        # Formater en HTML propre
        lines = sources_raw.replace("📚 **Sources :**", "").strip().split("\n")
        sources_items = "".join(
            f"<div>📄 {line.strip().lstrip('•').strip()}</div>"
            for line in lines if line.strip()
        )
        sources_html = f"""
        <div class='sources-block'>
            <strong>📚 Sources</strong>
            {sources_items}
        </div>
        """

        # ✅ Remplacer le placeholder avec réponse + sources
        with placeholder.chat_message("assistant"):
            st.write(answer_text)
            st.markdown(sources_html, unsafe_allow_html=True)
    else:
        answer_text = full_response

    # ✅ Sauvegarder réponse et sources séparément dans l'historique
    st.session_state.history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_html  # peut être "" si pas de sources
    })
