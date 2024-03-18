# Streamlit 
import streamlit as st

# langchain
import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Vector database
# from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Agents 
from agents_gemini import Obnoxious_Agent, Query_Agent, Relevant_Documents_Agent, Answering_Agent

# **Streamlit UI Elements**
st.title("Multi Agent RAG Chatbot: Gemini")

# Initialize Pinecone only once using caching
@st.cache_resource
def initialize_pinecone():
    pc = Pinecone()
    index = pc.Index('geminiembeddings')
    return index
    
index = initialize_pinecone()

# Initialize LLM and Vector store using caching to avoid reloading on every interaction
@st.cache_resource
def initialize_llm_embeddings():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = PineconeVectorStore(index=initialize_pinecone(),
                                      embedding=embeddings,
                                      text_key="text",
                                      namespace="embedding_1024_256")
    retriever = vectorstore.as_retriever()
    return llm, embeddings, vectorstore, retriever

llm, embeddings, vectorstore, retriever = initialize_llm_embeddings()

# Session State for API key and Chat history
if "is_valid" not in st.session_state:
    st.session_state["is_valid"] = True
if "messages" not in st.session_state:
    st.session_state['messages'] = []


class Head_Agent:
    def __init__(self, llm, embeddings, index, vectorstore, retriever) -> None:
        # llm + embeddings
        self.llm = llm
        self.embeddings = embeddings
        # Pinecone Index + vector store
        self.index = index
        self.vectorstore = vectorstore
        self.retriever = retriever
        # Filter user queries
        self.obnoxious_phrase = "Please do not ask inappropriate or obnoxious questions."
        self.non_relevant_phrase = "Please ask a question relevant to Machine Learning"
        # chat history prompt
        self.memory_prompt = """Given a chat history and the latest user question
        which might reference context in the chat history, formulate a standalone question
        which can be understood without the chat history. Do NOT answer the question,
        just reformulate it if needed and otherwise return it as is.
        Chat_History: {chat_history}
        """

    def setup_sub_agents(self):
        self.OA = Obnoxious_Agent(self.llm)
        self.QA = Query_Agent(self.vectorstore)
        self.AA = Answering_Agent(self.llm, self.vectorstore, self.retriever)
        self.RDA = Relevant_Documents_Agent(self.llm)
        print("Agents Initialized")

    def query_gemini(self, prompt) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error while checking with gemini: {e}")
            return None

    def set_chat_history(self, chat_history):
        self.memory_prompt = self.memory_prompt.format(chat_history=chat_history)

    def main_loop(self, query, chat_history):
      print("Query:", query)

      ## 1. Obnoxious Agent
      OA_prompt = self.OA.set_prompt(query)
      OA_action = self.OA.extract_action(OA_prompt)
      print("Obnoxious Agent Response:", OA_action)
      if OA_action:
        print("OA Message:", self.obnoxious_phrase)
        return self.obnoxious_phrase

      self.set_chat_history(chat_history)
      updated_prompt = self.query_gemini(self.memory_prompt)

      ## 2. Pinecone Query
      docs = self.QA.query_vector_store(updated_prompt)

      ## 3. Relevant
      RDA_prompt = self.RDA.set_prompt(docs, updated_prompt)
      RDA_response = self.RDA.get_relevance(RDA_prompt)
      print("Relevance Agent Response:", RDA_response)
      RDA_action = self.RDA.extract_action(RDA_response)
      if not RDA_action:
        print("RDA Message:", self.non_relevant_phrase)
        return self.non_relevant_phrase

      ## 4. Answering
      AA_prompt = self.AA.set_prompt(updated_prompt)
      AA_response = self.AA.generate_response(AA_prompt)
      return AA_response

# Initialize Head_Agent if the API key is valid
if st.session_state["is_valid"]:
    HA = Head_Agent(llm, embeddings, index, vectorstore, retriever)
    HA.setup_sub_agents()

    # Display existing chat messages
    for message in st.session_state['messages']:
        st.chat_message(message['role']).write(message['content'])

    # Main chat interaction loop
    if user_prompt := st.chat_input("What would you like to ask?"):
        st.session_state['messages'].append({"role": "user", "content": user_prompt})
        st.chat_message(st.session_state['messages'][-1]['role']).write(st.session_state['messages'][-1]['content'])

        print("Chat History:", st.session_state['messages'])
        response = HA.main_loop(query=user_prompt, chat_history=st.session_state['messages'])

        st.session_state['messages'].append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)    