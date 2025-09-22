import streamlit as st
import uuid
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from typing import Type
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
#from litellm import ChatLiteLLM


from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration ---
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
CREDS_FILE = "credentials.json"
SHEET_NAME = "ticketing_ai"

# Category-wise human agents
HUMAN_AGENTS = {
    "billing": ["Rahim", "Sara"],
    "technical": ["John", "Emily"],
    "network": ["Mike", "Lisa"],
    "general": ["David", "Sophia"]
}

# Dummy FAQ data
FAQ_DATA = {
    "internet speed packages": "We offer three internet speed packages: 10 Mbps, 50 Mbps, and 100 Mbps.",
    "working hours": "Our team is available from 9 AM to 6 PM, Monday to Friday.",
    "refund policy": "We offer a 7-day refund for any service downtime.",
    "contact number": "You can reach us at +1-800-555-1234.",
    "billing cycle": "Our billing cycle is from the 1st to the end of each month.",
    "payment methods": "We accept credit cards, debit cards, and bank transfers.",
    "service coverage": "We currently cover all major cities and surrounding areas.",
    "installation process": "Our technician will schedule an appointment within 2 business days.",
    "router setup": "We provide detailed setup instructions and 24/7 support for router configuration.",
    "speed test": "You can test your internet speed at speedtest.net or contact support for assistance."
}

# Initialize LLMs
def initialize_llms():
    """Initialize LLMs for GitHub models"""
    try:
        load_dotenv()

        BASE_URL = os.getenv("BASE_URL") 
        API_KEY = os.getenv("API_KEY") 
        MODEL_NAME = os.getenv("MODEL_NAME") 
        openai_api_key = os.getenv("OPENAI_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Validate that we have proper credentials
        if not API_KEY or not BASE_URL or not MODEL_NAME:
            st.error("Missing required environment variables: BASE_URL, API_KEY, or MODEL_NAME")
            return None, None
        #CrewAI expect ChatOPenAI interface
        # # For GitHub models, we need to use proper authentication
        openai_llm = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.1
        )

        # llm = ChatLiteLLM(
        #     model="meta/Llama-4-Scout-17B-16E-Instruct",
        #     provider="litellm_proxy",  # or another provider if applicable
        #     api_key=API_KEY,
        #     base_url="https://models.github.ai/inference",
        #     temperature=0.1
        # )

        # from google import genai
        # from langchain_google_genai import ChatGoogleGenerativeAI
        # client = genai.Client(api_key=gemini_api_key
        #                       ,Credentials=credentials.json)

        # geminillm = ChatGoogleGenerativeAI(
        #     model="gemini-pro",
        #     client=client,
        #     temperature=0.1
        #     )
        
        # response = client.models.generate_content(
        #     model="gemini-2.5-flash", 
        #     contents="Explain how AI works in a few words"
        # )
        # print(response.text)


        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini/gemini-1.5-flash", # You can also use "gemini-1.5-flash" or other compatible models
            google_api_key=gemini_api_key,
            temperature=0.1,
            # This is often helpful for compatibility with agent frameworks
            convert_system_message_to_human=True 
        )
        # ollama_llm = Ollama(
        #     model="gemma3:27b",
        #     temperature=0.1
        # )

        # openai_llm = ChatOpenAI(
        #     api_key=openai_api_key,
        #     model_name="gpt-4.1-mini",  # or whatever model you want to use
        #     temperature=0.1
        # )
        #to test openai
        # response = client.responses.create(
        # model="gpt-5-nano",
        # input="write a haiku about ai",
        # store=True,
        # )

        # print(response.output_text);

        #return llm, llm
        #return openai_llm, openai_llm
        return gemini_llm, gemini_llm
        
    except Exception as e:
        st.error(f"Error initializing LLMs: {e}")
        return None, None

#---------------------------------------Initialize  LLM- done--------------------------------------


# --- Guardrails with Ollama ---
def input_guardrail_ollama(query, ollama_llm):
    """
    Filters incoming requests using Ollama LLM to ensure they are within scope.
    """
    try:
        prompt = f"""
        Analyze the following customer query and determine if it's appropriate for customer support.
        If it contains topics like politics, medical advice, harmful content, elections, or violence, reject it.
        
        Query: "{query}"
        
        Respond with only "APPROVED" or "REJECTED" followed by a brief reason.
        """
        
        response = ollama_llm.invoke(prompt)
        response_text = str(response).strip()

        print("Guardrail Response:", response_text)  # Debugging line
        
        if "APPROVED" in response_text.upper():
            print("Guardrail Check: APPROVED")  # Debugging line
            return True, None
        else:
            print("Guardrail Check: REJECTED")  # Debugging line
            return False, "I can only assist with questions about our company's services."
            
    except Exception as e:
        print("Error in guardrail check:", e)
        # Fallback to keyword-based guardrail
        out_of_scope_keywords = ["politics", "medical advice", "harmful", "election", "violence"]
        for keyword in out_of_scope_keywords:
            if keyword in query.lower():
                return False, "I can only assist with questions about our company's services."
        return True, None

# Output guardrail to validate AI responses
def output_guardrail(response_text):
    """
    Validates AI responses to ensure they are appropriate and safe.
    """
    # List of inappropriate content indicators
    inappropriate_keywords = [
        "sorry, i cannot", "i'm not able to", "as an ai", "i am an ai",
        "illegal", "harmful", "dangerous", "inappropriate"
    ]
    
    # Check for empty or very short responses
    if not response_text or len(response_text.strip()) < 5:
        return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    # Check for inappropriate content
    response_lower = response_text.lower()
    for keyword in inappropriate_keywords:
        if keyword in response_lower:
            return "I'm sorry, I can only assist with questions about our company's services."
    
    # Check if response seems like an error message
    if "error" in response_lower and ("api" in response_lower or "key" in response_lower):
        return "I'm experiencing technical difficulties. Please try again later or contact our support team."
    
    return response_text




# --- Complaint Identification with OpenAI ---
def identify_question_type(query, openai_llm):
    """
    Use OpenAI to identify if the query is a problem statement or complaint or issue and categorize it.
    """
    try:
        prompt = f"""
        Analyze this customer query and categorize it:
        - If it's a FAQ question about services, respond with "FAQ"
        - If it's a general conversation not related to issues, respond with "GENERAL"
        - If it describes a problem or complaint, respond with "COMPLAINT"
        
        Available FAQ topics: {list(FAQ_DATA.keys())}
        
        Query: "{query}"
        
        Respond with only the category: "FAQ", "GENERAL", "COMPLAINT"
        """
        
        response = openai_llm.invoke(prompt)
        responsetext = str(response).strip().lower()

        print("Question Type Response:", responsetext)  # Debugging line
        if "FAQ" in responsetext.upper():
            type= "faq"
        elif "GENERAL" in responsetext.upper():
            type= "general"
        elif "COMPLAINT" in responsetext.upper():
            type= "complaint"
        else:
            type= "general"  # Default to general if unclear
        # Validate category
        #valid_categories = ["faq", "general", "billing", "technical", "network"]
        #return type if type in valid_categories else "general"
        return type

    except Exception as e:
        # Fallback to keyword-based categorization
        complaint_keywords = ["complaint", "problem", "issue", "not working", "broken", "error"]
        if any(keyword in query.lower() for keyword in complaint_keywords):
            return "complaint"
        elif any(keyword in query.lower() for keyword in FAQ_DATA.keys()):
            return "faq"
        else:
            return "general"

# --- CrewAI Setup for Advanced Processing ---
def setup_crewai_agents(openai_llm):
    """Setup CrewAI agents for advanced complaint processing"""
    
    # Create tool instance (without internal RAG reference)
    knowledge_tool = KnowledgeBaseTool()
    
    # Complaint Analyzer Agent
    complaint_category_analyzer = Agent(
        role='Complaint Analysis Specialist',
        goal='Analyze customer complaints and categorize them accurately',
        backstory='Expert in understanding customer issues and assigning them to the right category',
        llm=openai_llm,        
        verbose=True
    )

    # Complaint Severity Analyzer Agent
    complaint_severity_analyzer = Agent(
        role='Complaint Severity Specialist',
        goal='Analyze customer complaints and determine their severity',
        backstory='Expert in assessing the severity of customer issues',
        llm=openai_llm,
        verbose=True
    )
    
    # FAQ Specialist Agent
    faq_specialist = Agent(
        role='FAQ Knowledge Expert',
        goal='Provide accurate answers from FAQ database',
        backstory='Expert in company policies and service information',
        tools=[knowledge_tool],
        llm=openai_llm,
        verbose=True
    )
    
    return complaint_category_analyzer, complaint_severity_analyzer, faq_specialist
# --- Tools ---
# def knowledge_base_tool(query, faq_specialist):
#     """
#     Fetches answers from FAQ data using CrewAI agent.
#     """
#     try:
#         # Use CrewAI to answer from FAQ
#         print("Invoking knowledge_base_tool with query:", query)  # Debugging line
#         task = Task(
#             description=f"Answer this customer query using only the FAQ database: {query}",
#             tools=[knowledge_base_tool],
#             agent=faq_specialist,
#             expected_output="Concise, accurate answer from FAQ data only"
#         )
        
#         crew = Crew(
#             agents=[faq_specialist],
#             tasks=[task],
#             verbose=True
#         )
        
#         result = crew.kickoff()
#         print("Knowledge Base Tool Result:", result)  # Debugging line
#         return str(result)
#     except Exception as e:
#         # Fallback to simple FAQ lookup
#         for key, value in FAQ_DATA.items():
#             if key in query.lower():
#                 return value
#         return "I'm sorry, I don't have information on that topic."
    
#---------------RAG knowledge base tool-------------------
# class RAGKnowledgeBase:
#     def __init__(self):
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.client = chromadb.Client(Settings(anonymized_telemetry=False))
#         self.collection = self.client.get_or_create_collection("faq_knowledge_base")
#         self._initialize_faq_data()
    
#     def _initialize_faq_data(self):
#         """Initialize the FAQ data in ChromaDB"""
#         if self.collection.count() == 0:
#             documents = []
#             metadatas = []
#             ids = []
            
#             for i, (question, answer) in enumerate(FAQ_DATA.items()):
#                 documents.append(f"Question: {question}\nAnswer: {answer}")
#                 metadatas.append({"question": question, "answer": answer})
#                 ids.append(str(i))
            
#             # Add documents to collection
#             self.collection.add(
#                 documents=documents,
#                 metadatas=metadatas,
#                 ids=ids
#             )
    
#     def query(self, query_text, n_results=3):
#         """Query the knowledge base"""
#         try:
#             # Generate embedding for the query
#             query_embedding = self.embedding_model.encode(query_text).tolist()
            
#             # Search in ChromaDB
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=n_results
#             )
            
#             if results['documents'] and results['documents'][0]:
#                 return results['documents'][0][0]  # Return the most relevant document
#             else:
#                 return "I'm sorry, I don't have information on that topic."
                
#         except Exception as e:
#             print(f"Error querying knowledge base: {e}")
#             # Fallback to simple lookup
#             query_lower = query_text.lower()
#             for key, value in FAQ_DATA.items():
#                 if key in query_lower:
#                     return value
#             return "I'm sorry, I don't have information on that topic."

# Update the KnowledgeBaseTool to use RAG
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticKnowledgeBase:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._precompute_embeddings()
        except Exception as e:
            print(f"Error initializing semantic search: {e}")
            self.fallback_mode = True
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all FAQ items"""
        self.faq_questions = list(FAQ_DATA.keys())
        self.faq_answers = list(FAQ_DATA.values())
        
        # Generate embeddings for all questions
        self.question_embeddings = self.embedding_model.encode(self.faq_questions)
    
    def query(self, query_text, threshold=0.5):
        """Semantic search without ChromaDB"""
        try:
            if hasattr(self, 'fallback_mode') and self.fallback_mode:
                return self._fallback_query(query_text)
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity > threshold:
                return self.faq_answers[best_idx]
            else:
                return self._fallback_query(query_text)
                
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return self._fallback_query(query_text)
    
    def _fallback_query(self, query_text):
        """Fallback to keyword matching"""
        query_lower = query_text.lower()
        for key, value in FAQ_DATA.items():
            if key in query_lower:
                return value
        
        for key, value in FAQ_DATA.items():
            if any(word in query_lower for word in key.split()):
                return value
        
        return "I'm sorry, I don't have information on that topic."

# Global instance
semantic_kb = SemanticKnowledgeBase()

class KnowledgeBaseTool(BaseTool):
    name: str = "Knowledge Base Tool"
    description: str = "Fetches answers from FAQ database using semantic search"
    
    def _run(self, query: str) -> str:
        """
        Fetches answers from FAQ data using semantic search.
        """
        return semantic_kb.query(query)
#---------------------------------------------------------

def complain_analysis(complaint_text, user_name, category, complaint_category_analyzer,complaint_severity_analyzer):
    """
    Handles customer complaints with category-based agent assignment.
    """
    try:
        # Use CrewAI for detailed analysis
        task_severity = Task(
            description=f"Analyze this complaint and determine severity: {complaint_text}",
            agent=complaint_severity_analyzer,
            expected_output="Severity level: Low, Medium, or High"
        )
        task_category = Task(
            description=f"Analyze this complaint and determine category: {complaint_text}",
            agent=complaint_category_analyzer,
            expected_output="Category: Technical, Billing, or General"
        )

        severitycrew = Crew(
            agents=[complaint_severity_analyzer],
            tasks=[task_severity],
            verbose=True
        )
        categorycrew = Crew(
            agents=[complaint_category_analyzer],
            tasks=[task_category],
            verbose=True
        )
        category_result = categorycrew.kickoff()
        severity_result = severitycrew.kickoff()
        severity = "Medium"  # Default
        if "high" in str(severity_result).lower():
            severity = "High"
        elif "low" in str(severity_result).lower():
            severity = "Low"

        category = str(category_result).strip().lower()
        if category not in ["billing", "technical", "network"]:
            category = "general"
        print("Complaint Analysis Result - Category:", category, "Severity:", severity)  # Debugging line    
    except Exception as e:
        # Fallback severity determination
        severity = "Low"
        if "frustrated" in complaint_text.lower() or "down for days" in complaint_text.lower():
            severity = "High"
        elif "slow" in complaint_text.lower() or "intermittent" in complaint_text.lower():
            severity = "Medium"

    # Assign agent based on category
    agents = HUMAN_AGENTS.get(category, HUMAN_AGENTS["general"])
    ticket_id = f"TICKET-{uuid.uuid4().hex[:6].upper()}"
    assigned_agent = agents[hash(ticket_id) % len(agents)]

    return {
        "ticket_id": ticket_id,
        "complaint_text": complaint_text,
        "severity": severity,
        "user_name": user_name,
        "assigned_agent": assigned_agent,
        "category": category
    }

def google_sheet_tool(complaint_details):
    """
    Logs complaint information in Google Sheet.
    """
    try:
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
        client = gspread.authorize(creds)
        
        try:
            spreadsheet = client.open(SHEET_NAME)
        except gspread.SpreadsheetNotFound:
            st.error("Google Sheet not found. Please check the sheet name and sharing permissions.")
            return False
            
        sheet = spreadsheet.sheet1
        
        # Check if headers exist
        if not sheet.get_all_values():
            headers = ["Ticket ID", "User Name", "Complaint Text", "Severity", "Category", "Assigned Agent", "Timestamp"]
            sheet.append_row(headers)
        
        # Prepare the row data
        row_data = [
            complaint_details["ticket_id"],
            complaint_details["user_name"],
            complaint_details["complaint_text"],
            complaint_details["severity"],
            complaint_details["category"],
            complaint_details["assigned_agent"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        
        sheet.append_row(row_data)
        return True
        
    except Exception as e:
        st.error(f"Error logging to Google Sheet: {e}")
        return False

# --- Main Agent Logic ---
def customer_support_agent(user_query, user_name, ollama_llm, openai_llm, complaint_category_analyzer, complaint_severity_analyzer, faq_specialist):
    """
    Main function that orchestrates the AI agent's workflow.
    """
    try:
        # Input guardrail
        is_in_scope, message = input_guardrail_ollama(user_query, ollama_llm)
        if not is_in_scope:
            return message
        
        print("Input passed guardrail check.")
        
        # Identify query type
        query_type = identify_question_type(user_query, openai_llm)
        print("Identified type:", query_type)
        
        if query_type.upper() == "FAQ":
            # Use the FAQ specialist agent directly
            task = Task(
                description=f"Answer this customer query: {user_query}",
                agent=faq_specialist,
                expected_output="Concise, accurate answer"
            )
            
            crew = Crew(
                agents=[faq_specialist],
                tasks=[task],
                verbose=False
            )
            
            result = crew.kickoff()
            response = str(result)
            
        elif query_type.upper() == "GENERAL":
            # General conversation
            response = "I'm sorry, I can only assist with specific questions about our services or help with issues."
            
        else:
            # It's a complaint - create ticket
            complaint_details = complain_analysis(user_query, user_name, query_type, complaint_category_analyzer, complaint_severity_analyzer)
            if google_sheet_tool(complaint_details):
                response = (
                    f"Your {query_type} complaint has been logged as {complaint_details['severity']} priority. "
                    f"Agent {complaint_details['assigned_agent']} (specializing in {query_type}) will follow up shortly. "
                    f"Your ticket ID is {complaint_details['ticket_id']}."
                )
            else:
                response = "I'm sorry, there was an error logging your complaint. Please try again later."
        
        # Apply output guardrail
        return output_guardrail(response)
        
    except Exception as e:
        print(f"Error in customer_support_agent: {e}")
        return "I'm experiencing technical difficulties. Please try again later or contact our support team."

def safe_llm_call(llm_function, *args, **kwargs):
    """Safe wrapper for LLM calls with fallback"""
    try:
        return llm_function(*args, **kwargs)
    except Exception as e:
        print(f"LLM call failed: {e}")
        # Implement fallback logic here
        return "general"  # Default fallback

# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="Customer Support AI Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "llms_initialized" not in st.session_state:
        st.session_state.llms_initialized = False
    if "agents_initialized" not in st.session_state:
        st.session_state.agents_initialized = False
    
    # Sidebar for user name and info
    with st.sidebar:
        st.title("🤖 Customer Support")
        st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
        
        st.divider()
        st.subheader("About")
        st.write("This AI agent can help you with:")
        st.write("• FAQ questions about our services")
        st.write("• Technical support issues")
        st.write("• Billing and account questions")
        st.write("• Network problems")
        
        st.divider()
        st.subheader("Available Agents by Category")
        # for category, agents in HUMAN_AGENTS.items():
        #     st.write(f"**{category.title()}**: {', '.join(agents)}")
    
    # Main chat area
    st.title("Customer Support AI Agent")
    st.write("How can I help you today?")
    
    # Initialize LLMs and agents
    if not st.session_state.llms_initialized:
        with st.spinner("Initializing AI models..."):
            ollama_llm, openai_llm = initialize_llms()
            if ollama_llm and openai_llm:
                st.session_state.ollama_llm = ollama_llm
                st.session_state.openai_llm = openai_llm
                st.session_state.llms_initialized = True
            else:
                st.error("Failed to initialize AI models. Please check your API keys.")
                return
    
    if not st.session_state.agents_initialized and st.session_state.llms_initialized:
        with st.spinner("Setting up AI agents..."):
            complaint_category_analyzer, complaint_severity_analyzer, faq_specialist = setup_crewai_agents(st.session_state.openai_llm)
            st.session_state.complaint_category_analyzer = complaint_category_analyzer
            st.session_state.complaint_severity_analyzer = complaint_severity_analyzer
            st.session_state.faq_specialist = faq_specialist
            st.session_state.agents_initialized = True
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.user_name:
            st.warning("Please enter your name in the sidebar first.")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = customer_support_agent(
                    prompt, 
                    st.session_state.user_name,
                    st.session_state.ollama_llm,
                    st.session_state.openai_llm,
                    st.session_state.complaint_category_analyzer,
                    st.session_state.complaint_severity_analyzer,
                    st.session_state.faq_specialist
                )
                st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

