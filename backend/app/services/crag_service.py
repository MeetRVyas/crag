import json
import logging
from typing import List, TypedDict, Literal, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

from app.config import settings
from app.models.crag import Score, KeepOrDrop, WebQuery

load_dotenv()

# TODO : Add logging

# Graph state
# Passed between graph nodes
class State(TypedDict):
    # Inputs
    question: str
    
    # Retrieval Data
    hypothetical_answer: str
    docs: List[Document]       # Initial retrieved docs
    
    # Evaluation Data
    good_docs: List[Document]  # Docs deemed relevant by evaluator
    verdict: str               # "CORRECT", "INCORRECT", "AMBIGUOUS"
    reason: str                # Why this verdict?
    
    # Web Search Data
    web_query: str
    web_docs: List[Document]
    
    # Refinement Data
    strips: List[str]          # Decomposed sentences
    kept_strips: List[str]     # Filtered sentences
    refined_context: str       # Final context string
    
    # Output
    answer: str


# Corrective RAG pipeline using LangGraph
class CRAG_Service :
    # Thresholds
    UPPER_THRESHOLD = 0.7
    LOWER_THRESHOLD = 0.3

    def __init__(
            self,
            session_id : int,
            retriever,
            redis_client,
            db_session,
            provider = "ollama",
            api_keys = None
        ) :
        self.session_id = session_id
        self.retriever = retriever
        self.redis = redis_client
        self.db = db_session
        self.api_keys = api_keys or {}

        self.llm = self._initialize_llm(provider)
        self.tavily = self._initialize_tavily()
        self.app = self._build_graph()
    
    def _initialize_llm(self, provider : str) :
        """
        This function :
            - Initializes LLM from the selected provider
            - Validates the API key where required
        """
        # TODO : Validate that the provider supports the model
        if provider == "ollama" :
            return ChatOllama(
                model = settings.LLM_MODEL,
                temperature = 0,
                base_url = "http://ollama:11434"
            )
        elif provider == "google" :
            api_key = self.api_keys.get(provider)
            if not api_key :
                raise ValueError("Google API key not found")
            
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                google_api_key=api_key
            )
        # TODO : Implement more providers
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _initialize_tavily(self):
        """
        This function :
            - Initializes Tavily web search tool
            - Validates the Tavily API key
        """
        api_key = self.api_keys.get("tavily")
        
        if not api_key:
            # We return None here and handle it gracefully in the node
            # Don't raise a error on initialization
            # as the users have a choice to disable web search
            # TODO : Log a warning here
            return None
        
        return TavilySearch(
            max_results = 5,
            topic = settings.TOPIC
        )
    
    def _build_graph(self) :
        """Defines the nodes and edges of the LangGraph state machine"""
        workflow = StateGraph(State)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("evaluate", self.evaluate)
        workflow.add_node("rewrite", self.rewrite)
        workflow.add_node("research", self.research)
        workflow.add_node("refine", self.refine)
        workflow.add_node("generate", self.generate)

        workflow.add_edge(START, 'retrieve')
        workflow.add_edge('retrieve', "evaluate")

        def route(state : State) :
            if state["verdict"] == "CORRECT" :
                return "refine"
            return "rewrite"
        
        workflow.add_conditional_edges(
            "evaluate",
            route,
            {
                "refine" : "refine",
                "rewrite" : "rewrite"
            }
        )

        workflow.add_edge('rewrite', "research")
        workflow.add_edge("research", "refine")
        workflow.add_edge('refine', "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, question : str) :
        self.app.invoke({"question" : question})

    def retrieve(self, state : State) -> Dict[str, Any] :
        """Node 1 : Retrieve documents using HyDE *Hypothetical Document Embeddings approach"""
        question = state["question"]

        # HyDE Approach
        # The user's query (question) might not be enough to answer the user's question
        # So we ask LLM to give a hypothetical answer without any context or facts
        hyde_prompt = ChatPromptTemplate.from_template(
            "You are an expert. Write a detailed, technical paragraph answering this question: {question}. "
            "Do not verify facts, just write a plausible answer."
        )
        hyde_chain = hyde_prompt | self.llm
        hypothetical = hyde_chain.invoke({"question": question})
        # Handle different return types from different LLM wrappers
        # OllamaLLM returns str whereas ChatGoogleGenerativeAI returns a response object
        hypothetical_text = hypothetical.content if hasattr(hypothetical, 'content') else str(hypothetical)

        # Retrieve using the hypothetical answer
        docs = self.retriever.invoke(hypothetical_text)

        return {
            "hypothetical_answer": hypothetical_text,
            "docs": docs
        }
    
    def evaluate(self, state: State) -> Dict[str, Any]:
        """Node 2 : Scores each document 0-1"""
        question = state["question"]
        docs = state["docs"]

        # Setup scoring chain
        parser = PydanticOutputParser(pydantic_object = Score)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict grader, checking if a document is relevant to a question.\n"
                       "Return a score between 0.0 and 1.0.\n"
                       "1.0 = Highly relevant/Direct answer. Document has the perfect answer.\n"
                       "0.0 = Irrelevant.\n"
                       f"> {self.UPPER_THRESHOLD} = Very Relevant. Document sufficient to answer the question.\n"
                       f"> {self.LOWER_THRESHOLD} = Some Relevance. Document necessary, but not enough to answer the question.\n"
                       "Be conservative.\n"
                       "Return JSOM : {format_instructions}"),
            ("human", "Question: {question}\n\nDocument:\n{context}")
        ])
        
        chain = prompt | self.llm | parser

        good_docs = []
        scores = []
        
        for doc in docs:
            try:
                res : Score = chain.invoke({
                    "question": question,
                    "context": doc.page_content,
                    "format_instructions": parser.get_format_instructions()
                })
                score = res.score
            except Exception as e :
                # TODO : Log a warning here
                score = 0.0

            scores.append(score)
            if score > self.LOWER_THRESHOLD:
                good_docs.append(doc)

        # Determine Verdict
        max_score = max(scores) if scores else 0.0
        
        if max_score > self.UPPER_THRESHOLD :
            verdict = "CORRECT"
        elif max_score > self.LOWER_THRESHOLD :
            verdict = "INCORRECT"
        else:
            verdict = "AMBIGUOUS"


        return {
            "good_docs": good_docs,
            "verdict": verdict,
            "reason": f"Max score: {max_score}"
        }
    
    def rewrite(self, state: State) -> Dict[str, Any]:
        """Node 3 : Rewrite question for Web Search"""
        question = state["question"]

        parser = PydanticOutputParser(pydantic_object = WebQuery)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Google Searcher.\n"
                       "Transform the user question into a targeted search query.\n"
                       "Strip unnecessary words. Add context if needed.\n"
                       "Return JSON : {format_instructions}"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | parser
        
        try :
            res : WebQuery = chain.invoke({
                "question": question,
                "format_instructions": parser.get_format_instructions()
            })
            query = res.query
        except Exception :
            # TODO : Log a warning here
            query = question # Fallback

        return {"web_query": query}

    def research(self, state: State) -> Dict[str, Any]:
        """Node 4: Search the Web"""
        query = state.get("web_query", state["question"])
        
        if not self.tavily:
            return {"web_docs": []}

        try:
            # Tavily returns a list of dicts
            results = self.tavily.invoke({"query": query})
            web_docs = []
            for r in results:
                web_docs.append(Document(
                    page_content=r.get("content", ""),
                    metadata={"source": r.get("url"), "title": r.get("title")}
                ))
            return {"web_docs": web_docs}
        except Exception as e:
            return {"web_docs": []}
    
    def _return_docs_based_on_verdict(self, state) -> List[Document] :
        verdict = state["verdict"]
        if verdict == "CORRECT":
            return state["good_docs"]
        elif verdict == "INCORRECT":
            return state["web_docs"]
        else: # AMBIGUOUS
            return state["good_docs"] + state["web_docs"]
    
    def _decompose_into_sentences(self, docs : List[Document]) -> str :
        import re
        full_text = " ".join([d.page_content for d in docs])
        # Simple regex split on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        return [s.strip() for s in sentences if len(s) > 20] # Filter tiny fragments
    
    def refine(self, state: State) -> Dict[str, Any]:
        """Node 5: Refine context"""
        # Select Source Documents based on Verdict
        docs = self._return_docs_based_on_verdict(state)
        
        # Decompose into sentences
        sentences = self._decompose_into_sentences(docs)
        
        # Filter sentences
        parser = PydanticOutputParser(pydantic_object=KeepOrDrop)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Read the following sentence and the user question.\n"
                       "Return 'keep': true if the sentence contains information relevant to the question.\n"
                       "Return 'keep': false if it is irrelevant, conversational filler, or meta-data.\n"
                       "{format_instructions}"),
            ("human", "Question: {question}\nSentence: {sentence}")
        ])
        chain = prompt | self.llm | parser
        
        kept = []
        # TODO : Remove sentences limit for production
        for sent in sentences[:20]: # Limit to first 20 sentences to save time/tokens for now
            try:
                res : KeepOrDrop = chain.invoke({
                    "question": state["question"],
                    "sentence": sent,
                    "format_instructions": parser.get_format_instructions()
                })
                if res.keep:
                    kept.append(sent)
            except:
                kept.append(sent) # Keep on error
                
        refined_context = "\n".join(kept)
        return {
            "strips": sentences,
            "kept_strips": kept,
            "refined_context": refined_context
        }

    def generate(self, state: State) -> Dict[str, Any]:
        """Node 6: Generate Final Answer"""
        question = state["question"]
        context = state["refined_context"]
        
        if not context:
            return {"answer": "I could not find enough relevant information to answer your question."}
            
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based ONLY on the following context.\n"
            "If the context does not answer the question, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )
        
        chain = prompt | self.llm
        response = chain.invoke({"question": question, "context": context})
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {"answer": answer}