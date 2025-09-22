import os
import uuid
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
import streamlit as st

# --- Configuration ---
# To use the Google Sheets API, you need to set up a service account and
# enable the Google Sheets and Google Drive APIs.
# For detailed instructions, refer to the gspread documentation:
# https://gspread.readthedocs.io/en/latest/oauth2.html
#
# Once you have your credentials, store them in a file named 'credentials.json'
# and place it in the same directory as this script.
#
# You will also need to share your Google Sheet with the client_email
# from your credentials.json file.

# Google Sheets configuration - CORRECTED SCOPES
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
CREDS_FILE = "credentials.json"
SHEET_NAME = "ticketing_ai"  # Make sure this matches your actual sheet name

# Predefined human agents for complaint assignment
HUMAN_AGENTS = ["Rahim", "Karim", "Toha"]

# Dummy FAQ data
FAQ_DATA = {
    "internet speed packages": "We offer three internet speed packages: 10 Mbps, 50 Mbps, and 100 Mbps.",
    "working hours": "Our team is available from 9 AM to 6 PM, Monday to Friday.",
    "refund policy": "We offer a 7-day refund for any service downtime.",
    "contact number": "You can reach us at +1-800-555-1234.",
    "billing cycle": "Our billing cycle is from the 1st to the end of each month.",
}

# --- Guardrails ---

def input_guardrail(query):
    """
    Filters incoming requests to ensure they are within the scope of customer support.
    """
    out_of_scope_keywords = ["politics", "medical advice", "harmful", "election", "violence"]
    for keyword in out_of_scope_keywords:
        if keyword in query.lower():
            return False, "I can only assist with questions about our company's services."
    return True, None

def output_guardrail(response):
    """
    Ensures the generated response is professional and safe.
    """
    if not response or not isinstance(response, str):
        return "I'm sorry, I couldn't generate a valid response."

    # Additional checks can be added here, such as for bias or harmful content.
    return response.strip()

# --- Tools ---

def knowledge_base_tool(query):
    """
    Fetches answers from the dummy FAQ data.
    """
    for key, value in FAQ_DATA.items():
        if key in query.lower():
            return value
    return "I'm sorry, I don't have information on that topic."

def complain_ticket_tool(complaint_text, user_name):
    """
    Handles customer complaints by categorizing severity and assigning to an agent.
    """
    severity = "Low"  # Default severity
    if "frustrated" in complaint_text.lower() or "down for days" in complaint_text.lower():
        severity = "High"
    elif "slow" in complaint_text.lower() or "intermittent" in complaint_text.lower():
        severity = "Medium"

    ticket_id = f"TICKET-{uuid.uuid4().hex[:6].upper()}"
    assigned_agent = HUMAN_AGENTS[hash(ticket_id) % len(HUMAN_AGENTS)]

    return {
        "ticket_id": ticket_id,
        "complaint_text": complaint_text,
        "severity": severity,
        "user_name": user_name,
        "assigned_agent": assigned_agent,
    }

def google_sheet_tool(complaint_details):
    """
    Logs complaint information in a Google Sheet.
    """
    try:
        # Use the correct scope for authentication
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1

        # Prepare the row data
        row_data = [
            complaint_details["ticket_id"],
            complaint_details["user_name"],
            complaint_details["complaint_text"],
            complaint_details["severity"],
            complaint_details["assigned_agent"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ]
        sheet.append_row(row_data)
        return True
    except Exception as e:
        print(f"Error logging to Google Sheet: {e}")
        return False

# --- Main Agent Logic ---

def customer_support_agent(user_query, user_name="Customer"):
    """
    The main function that orchestrates the AI agent's workflow.
    """
    is_in_scope, message = input_guardrail(user_query)
    if not is_in_scope:
        return output_guardrail(message)

    # Check if the query is a complaint
    complaint_keywords = ["complaint", "frustrated", "down", "not working", "slow","problem","issue"]
    is_complaint = any(keyword in user_query.lower() for keyword in complaint_keywords)

    if is_complaint:
        complaint_details = complain_ticket_tool(user_query, user_name)
        if google_sheet_tool(complaint_details):
            response = (
                f"Your complaint has been logged as {complaint_details['severity']} priority. "
                f"Agent {complaint_details['assigned_agent']} will follow up shortly. "
                f"Your ticket ID is {complaint_details['ticket_id']}."
            )
        else:
            response = "I'm sorry, there was an error logging your complaint. Please try again later."
    else:
        response = knowledge_base_tool(user_query)

    return output_guardrail(response)

# --- Example Usage ---




#============================
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
        # for  agents in HUMAN_AGENTS.items():
        #     st.write(f"**Complaint**: {', '.join(agents)}")
    
    # Main chat area
    st.title("Customer Support AI Agent")
    st.write("How can I help you today?")
    
    # Initialize LLMs and agents
    if not st.session_state.llms_initialized:
        with st.spinner("Initializing AI models..."):
            #ollama_llm, openai_llm = initialize_llms()
            # if ollama_llm and openai_llm:
            #     st.session_state.ollama_llm = ollama_llm
            #     st.session_state.openai_llm = openai_llm
            st.session_state.llms_initialized = True
            # else:
            #     st.error("Failed to initialize AI models. Please check your API keys.")
            #     return
    
    if not st.session_state.agents_initialized and st.session_state.llms_initialized:
        with st.spinner("Setting up AI agents..."):
            # complaint_category_analyzer, complaint_severity_analyzer, faq_specialist = setup_crewai_agents(st.session_state.openai_llm)
            # st.session_state.complaint_category_analyzer = complaint_category_analyzer
            # st.session_state.complaint_severity_analyzer = complaint_severity_analyzer
            # st.session_state.faq_specialist = faq_specialist
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
                    st.session_state.user_name
                    # st.session_state.ollama_llm,
                    # st.session_state.openai_llm,
                    # st.session_state.complaint_category_analyzer,
                    # st.session_state.complaint_severity_analyzer,
                    # st.session_state.faq_specialist
                )
                st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

#============================


if __name__ == "__main__":
    main()
    # print("Welcome to the Customer Support AI Agent!")
    # print("You can ask questions or file a complaint. Type 'exit' to quit.")

    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() == "exit":
    #         break

    #     # In a real-world scenario, you would get the user's name from their profile
    #     agent_response = customer_support_agent(user_input, user_name="John Doe")
    #     print(f"Agent: {agent_response}")