import streamlit as st
import uuid
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuration ---
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
CREDS_FILE = "credentials.json"
SHEET_NAME = "ticketing_ai"

load_dotenv()
openai_model = os.getenv("MODEL_NAME")

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

# Initialize OpenAI client
def initialize_llm():
    """Initialize OpenAI client"""
    try:
        load_dotenv()
        openai_api_key = os.getenv("API_KEY")

        if not openai_api_key:
            st.error("Missing required environment variable: OPENAI_API_KEY")
            return None, None
        
        #openai_client = OpenAI(api_key=openai_api_key)
        openai_client = OpenAI(
            api_key=openai_api_key,  # GitHub Personal Access Token
            base_url="https://models.github.ai/inference",  # GitHub Models endpoint
        )
        #ollama_llm = None 
        return openai_client, openai_client
        
    except Exception as e:
        st.error(f"Error initializing LLMs: {e}")
        return None, None

# --- Guardrails ---
def input_guardrail_openai(client, query):
    """
    Filters incoming requests using OpenAI LLM to ensure they are within scope.
    """
    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Analyze the user's query and determine if it's appropriate for customer support. If it contains topics like politics, medical advice, harmful content, elections, or violence, reject it. Respond with only 'APPROVED' or 'REJECTED' followed by a brief reason."},
                {"role": "user", "content": f"Query: \"{query}\""}
            ],
            temperature=0.1,
            max_tokens=50
        )
        response_text = response.choices[0].message.content.strip()
        print("Guardrail Response:", response_text)
        
        if "APPROVED" in response_text.upper():
            print("Guardrail Check: APPROVED")
            return True, None
        else:
            print("Guardrail Check: REJECTED")
            return False, "I can only assist with questions about our company's services."
            
    except Exception as e:
        print("Error in guardrail check:", e)
        out_of_scope_keywords = ["politics", "medical advice", "harmful", "election", "violence"]
        for keyword in out_of_scope_keywords:
            if keyword in query.lower():
                return False, "I can only assist with questions about our company's services."
        return True, None

def output_guardrail(response_text):
    """
    Validates AI responses to ensure they are appropriate and safe.
    """
    inappropriate_keywords = [
        "sorry, i cannot", "i'm not able to", "as an ai", "i am an ai",
        "illegal", "harmful", "dangerous", "inappropriate"
    ]
    
    if not response_text or len(response_text.strip()) < 5:
        return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    response_lower = response_text.lower()
    for keyword in inappropriate_keywords:
        if keyword in response_lower:
            return "I'm sorry, I can only assist with questions about our company's services."
    
    if "error" in response_lower and ("api" in response_lower or "key" in response_lower):
        return "I'm experiencing technical difficulties. Please try again later or contact our support team."
    
    return response_text

# --- Complaint Identification with OpenAI ---
def identify_question_type(client, query):
    """
    Use OpenAI to identify if the query is a problem statement or complaint.
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
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that categorizes customer queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        responsetext = response.choices[0].message.content.strip().upper()
        
        print("Question Type Response:", responsetext)
        if "FAQ" in responsetext:
            return "faq"
        elif "GENERAL" in responsetext:
            return "general"
        elif "COMPLAINT" in responsetext:
            return "complaint"
        else:
            return "general"
            
    except Exception as e:
        print(f"Error in identify_question_type: {e}")
        complaint_keywords = ["complaint", "problem", "issue", "not working", "broken", "error"]
        if any(keyword in query.lower() for keyword in complaint_keywords):
            return "complaint"
        else:
            return "general"

# --- RAG/Knowledge Base for FAQs ---
class SemanticKnowledgeBase:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        self.faq_questions = list(FAQ_DATA.keys())
        self.faq_answers = list(FAQ_DATA.values())
        self.question_embeddings = self.embedding_model.encode(self.faq_questions)
    
    def query(self, query_text, threshold=0.5):
        query_embedding = self.embedding_model.encode([query_text])
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity > threshold:
            return self.faq_answers[best_idx]
        else:
            return "I'm sorry, I don't have information on that topic."

# Global instance
semantic_kb = SemanticKnowledgeBase()

# --- Complaint Analysis & Ticket Creation without CrewAI ---
def analyze_complaint(client, complaint_text):
    """
    Analyzes a complaint using OpenAI to determine its category and severity.
    """
    try:
        prompt_category = f"Analyze the following customer complaint and determine its category. Respond with only one word: 'technical', 'billing', 'network', or 'general'. Complaint: '{complaint_text}'"
        response_category = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a specialist in categorizing customer complaints."},
                {"role": "user", "content": prompt_category}
            ],
            temperature=0.1,
            max_tokens=10
        ).choices[0].message.content.strip().lower()

        prompt_severity = f"Analyze the following customer complaint and determine its severity. Respond with only one word: 'low', 'medium', or 'high'. Complaint: '{complaint_text}'"
        response_severity = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a specialist in determining the severity of customer issues."},
                {"role": "user", "content": prompt_severity}
            ],
            temperature=0.1,
            max_tokens=10
        ).choices[0].message.content.strip().lower()

        category = re.sub(r'[^a-z]', '', response_category)
        severity = re.sub(r'[^a-z]', '', response_severity)

        if category not in ["billing", "technical", "network"]:
            category = "general"
        if severity not in ["low", "medium", "high"]:
            severity = "medium"

        return category, severity
    except Exception as e:
        print(f"Error in analysis: {e}")
        return "general", "medium" # Fallback

def complain_analysis(complaint_text, user_name, client):
    """
    Handles customer complaints, determines category/severity, and assigns an agent.
    """
    category, severity = analyze_complaint(client, complaint_text)

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
        
        if not sheet.get_all_values():
            headers = ["Ticket ID", "User Name", "Complaint Text", "Severity", "Category", "Assigned Agent", "Timestamp"]
            sheet.append_row(headers)
        
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
def customer_support_agent(user_query, user_name, client):
    """
    Main function that orchestrates the AI agent's workflow.
    """
    try:
        is_in_scope, message = input_guardrail_openai(client, user_query)
        if not is_in_scope:
            return message
        
        print("Input passed guardrail check.")
        
        query_type = identify_question_type(client, user_query)
        print("Identified type:", query_type)
        
        if query_type == "faq":
            response = semantic_kb.query(user_query)
        elif query_type == "general":
            response = "I'm sorry, I can only assist with specific questions about our services or help with issues."
        else:
            complaint_details = complain_analysis(user_query, user_name, client)
            if google_sheet_tool(complaint_details):
                response = (
                    f"Your {complaint_details['category']} complaint has been logged as {complaint_details['severity']} priority. "
                    f"Agent {complaint_details['assigned_agent']} will follow up shortly. "
                    f"Your ticket ID is {complaint_details['ticket_id']}."
                )
            else:
                response = "I'm sorry, there was an error logging your complaint. Please try again later."
        
        return output_guardrail(response)
        
    except Exception as e:
        print(f"Error in customer_support_agent: {e}")
        return "I'm experiencing technical difficulties. Please try again later or contact our support team."

# --- Streamlit App ---
def main():
    st.set_page_config(
        page_title="Customer Support AI Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False
    
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
    
    st.title("Customer Support AI Agent")
    st.write("How can I help you today?")
    
    if not st.session_state.llm_initialized:
        with st.spinner("Initializing AI models..."):
            client, _ = initialize_llm()
            if client:
                st.session_state.openai_client = client
                st.session_state.llm_initialized = True
            else:
                st.error("Failed to initialize AI models. Please check your API keys.")
                return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.user_name:
            st.warning("Please enter your name in the sidebar first.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = customer_support_agent(
                    prompt, 
                    st.session_state.user_name,
                    st.session_state.openai_client
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()