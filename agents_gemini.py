# langchain
import langchain
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

class Obnoxious_Agent:
    """
    Use:
        OB = Obnoxious_Agent(llm)
        OB.set_prompt("How are you?")
        response = OB.extract_action()
    """
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = """
            Answer STRICTLY with "Yes" or "No".
            As an AI trained in understanding human language nuances and emotions,
            you are tasked with evaluating the sentiment and appropriateness of the text provided below.
            Please analyze the tone, the specific words and phrases used, and the overall context.
            Your goal is to discern whether the text comes across as obnoxious, rude, or inappropriate in any way.
            After your analysis, respond with \"Yes\" if you find the text to be generally obnoxious or inappropriate,
            and \"No\" if the text seems generally acceptable and not obnoxious. "
            Keep in mind the subtleties of human communication and consider the potential for misunderstanding or misinterpretation.
            context: {context}
            """

    def set_prompt(self, prompt=None):
        self.prompt = self.prompt.format(context=prompt)
        return self.prompt

    def extract_action(self, prompt) -> bool:
        try:
            response = self.llm.invoke(prompt)
            if response:
                return False
        except Exception as e:  # Catch any potential exceptions
            return True  # Default response: Not obnoxious

class Query_Agent:
  """
  Use:  
      QA = Query_Agent(vectorstore)
      documents = QA.query_vector_store("What are decision trees?")
  """
  def __init__(self, vectorstore) -> None:
    self.vectorstore = vectorstore
    self.documents = []

  def query_vector_store(self, query, k=3):
    self.documents = [x.page_content for x in self.vectorstore.similarity_search(query)]
    return self.documents
  
class Relevant_Documents_Agent:
  def __init__(self, llm) -> None:
    self.llm = llm
    self.prompt = """[INST]Answer STRICTLY with "Yes" or "No". 
    This is important: If the user querie or the question below is a general greeting such as "hello" then you must reply with "Yes".
    Use the following context to check if the query is relevant or not.
    If the context is even slightly relevant then reply with "Yes" and 
    if the context and query are poles apart then reply with "No".
    Context: {context}
    Question: {query} [/INST]
    This is important: If the user querie or the question below is a general greeting such as "hello" then you must reply with "Yes".
    """
    self.query = ""

  def set_prompt(self, documents, query):
    self.prompt = self.prompt.format(context=documents, query=query)
    return self.prompt

  def get_relevance(self, prompt) -> str:
    try:
      response = self.llm.invoke(prompt)
      return response.content
    except Exception as e:
      print(f"Error while checking relevance: {e}")
      return None

  def extract_action(self, response = None):
    if response == "Yes":
      return True
    else:
      return False

class Answering_Agent:
  """
  Use:
      AA = Answering_Agent(llm, vectorstore, retriever)
      AA_prompt = AA.set_prompt("What are decision trees?")
      response = AA.generate_response()
  """
  def __init__(self, llm, vectorstore, retriever) -> None:
      self.llm = llm
      self.vectorstore = vectorstore
      self.retriever = retriever
      self.prompt_template = ChatPromptTemplate.from_template("""As an AI, you are provided with relevant pieces of 
                                                              information to help answer the following user query.
                                                              Utilize the insights from these texts to formulate a 
                                                              comprehensive and accurate response. Your goal is to 
                                                              synthesize the information, highlight key points, and 
                                                              ensure the answer is informative and directly addresses 
                                                              the user's question. You will also be given the previous 
                                                              history of chat as Context use it to influence the answer.
                                                              This is important: If the user queries with any kind of general 
                                                              greeting such as "hello", respond with a general greeting.
                                                              Relevant Text : {relevant_text}
                                                              User's Query: {query}
                                                              """)
      self.chain =  (
                    {
                      "relevant_text": itemgetter("query")| self.retriever,
                      "query": RunnablePassthrough(),
                    }
                    | self.prompt_template
                    | self.llm
                    | StrOutputParser()
                    )
      self.prompt = {}

  def set_prompt(self, query):
    self.prompt = {"relevant_text": query,
                   "query": query}
    return self.prompt

  def generate_response(self, prompt):
      return self.chain.invoke(prompt)
