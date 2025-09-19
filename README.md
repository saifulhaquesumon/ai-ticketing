# ai-ticketing
Raise and assign ticket

Ethically Safe AI Agent with Guardrails
This project is a customer support AI agent designed with a strong focus on ethical and safe user interactions. The agent is equipped with input and output guardrails to ensure it operates within its designated scope, providing professional and relevant responses. It can answer frequently asked questions, handle customer complaints, and log them into a Google Sheet for further action.

🚀 Live Demo & Resources
Google Sheet Link: View a read-only version of the complaint log here.

Demo Video: Watch a 2-minute demo of the agent in action.

Colab Link: Run the agent directly in Google Colab.

✨ Features
Ethical Guardrails: Ensures the agent only responds to relevant and safe queries.

FAQ Handling: The knowledge_base_tool provides instant answers to common questions.

Complaint Management: The complain_ticket_tool processes and categorizes user complaints.

Google Sheets Integration: The google_sheet_tool logs all complaints for tracking and resolution.

Predefined Agents: A team of human agents is ready to handle escalated issues.

🛠️ How It Works
Ethical Guardrails
The agent's ethical framework is built on two key components:

Input Guardrails: Before processing any query, the input guardrail function checks if the user's request is within the scope of customer support. It filters out questions related to sensitive topics such as politics, medical advice, or any harmful content. If a query is out of scope, the agent politely declines to answer.

Output Guardrails: After generating a response, the output guardrail function reviews it to ensure it is professional, unbiased, and free from any irrelevant or harmful information. This guarantees that all communication from the agent aligns with the company's standards.

Tools
The agent utilizes three specialized tools to manage user interactions:

knowledge_base_tool: This tool contains a predefined set of frequently asked questions and their answers. When a user asks a question about the company's services, this tool is used to fetch the relevant information, providing a quick and accurate response.

complain_ticket_tool: When a user expresses a complaint, this tool is activated. It analyzes the complaint to determine its severity (Low, Medium, or High), generates a unique ticket ID, and assigns it to a human agent from a predefined list.

google_sheet_tool: After a complaint ticket is created, this tool logs all the relevant details—including the ticket ID, user's name, complaint description, severity, and the assigned agent—into a Google Sheet. This creates a persistent record that the support team can use to track and resolve issues.

Workflow
The user sends a query to the agent.

The input guardrail assesses the query for relevance and safety.

If the query is a general question, the knowledge_base_tool is used.

If the query is a complaint, the complain_ticket_tool categorizes it and assigns it to an agent.

The google_sheet_tool logs the complaint details.

The output guardrail ensures the final response is appropriate.

The user receives a confirmation with the ticket ID and the name of the assigned agent.

Example Conversations
Scenario 1: Asking a question

User: "What are your working hours?"

Agent: "Our team is available from 9 AM to 6 PM, Monday to Friday."

Scenario 2: Making a complaint

User: "My internet has been down for three days, and I'm very frustrated."

Agent: "I'm sorry to hear that. Your complaint has been logged as High priority. Agent Sara will follow up with you shortly. Your ticket ID is #XYZ-123."

Scenario 3: Asking an out-of-scope question

User: "What is your opinion on the upcoming election?"

Agent: "I can only assist with questions about our company's services. Please let me know if there is anything else I can help you with."