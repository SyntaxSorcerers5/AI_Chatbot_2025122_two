import os
import json
from datetime import datetime
from fpdf import FPDF
from rapidfuzz import fuzz, process
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from nltk_utils import tokenize
from model import NeuralNet
import torch
from nltk_utils import preprocess_input, fuzzy_match_with_synonyms
from fpdf import FPDF
from datetime import datetime
import torch
import torch.nn as nn

# OpenAI GPT-4 API Configuration
openai_api_key = "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"

# Load knowledge base
knowledge_base_path = 'knowledge_base.json'
if not os.path.exists(knowledge_base_path):
    raise FileNotFoundError(f"Error: {knowledge_base_path} not found.")

with open(knowledge_base_path, 'r') as f:
    knowledge_base = json.load(f)

# Initialize LangChain's ChatOpenAI
chat_model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

# Persistent context memory for LangChain
context_memory = []

# Directory to save conversations
conversation_dir = 'conversations'
os.makedirs(conversation_dir, exist_ok=True)

# Path for new solutions file
new_solutions_path = 'new_solution.json'

# Load model data
model_data = torch.load('data.pth')

# Load model
model = NeuralNet(model_data['input_size'], model_data['hidden_size'], model_data['output_size'])
model.load_state_dict(model_data['model_state'])
model.eval()

def save_new_solution_to_file(question, answer):
    """Save new information to the new_solution.json file."""
    new_solutions = {}
    if os.path.exists(new_solutions_path):
        try:
            with open(new_solutions_path, 'r') as f:
                new_solutions = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {new_solutions_path}. Initializing a new file.")

    new_solutions[question.lower()] = answer
    with open(new_solutions_path, 'w') as f:
        json.dump(new_solutions, f, indent=4)

def fuzzy_match_knowledge_base(msg, threshold=80):
    """Fuzzy match user input with the knowledge base keys."""
    preprocessed_msg = preprocess_input(msg)
    return fuzzy_match_with_synonyms(preprocessed_msg, knowledge_base.keys(), threshold)

def get_first_aid_details(accident_type, severity):
    """Retrieve first aid instructions for the given accident and severity."""
    severity = severity.lower()  # Normalize user input to lowercase
    if accident_type in knowledge_base:
        # Normalize keys in knowledge base to lowercase for matching
        for key, value in knowledge_base[accident_type].items():
            if key.lower() == severity:
                return value
    return "No specific instructions available for this case."

def get_response(msg):
    """Generate a response using the knowledge base or GPT model."""
    global context_memory

    # Use fuzzy matching to retrieve from the knowledge base
    matched_key = fuzzy_match_knowledge_base(msg)
    if matched_key:
        severity = "minor"  # Default to minor severity
        # Ask severity follow-up questions based on the matched accident type
        if matched_key == 'fainting':
            severity_input = input("Was the unconsciousness more than one minute? (Yes/No): ").strip().lower()
            severity = 'serious' if severity_input == 'yes' else 'minor'
        elif matched_key == 'cut':
            severity_input = input("Is the wound shallow or deep? (Shallow/Deep): ").strip().lower()
            severity = 'minor' if severity_input == 'shallow' else 'serious'
        elif matched_key == 'nosebleed':
            severity_input = input("Has the bleeding lasted more than 10 minutes? (Yes/No): ").strip().lower()
            severity = 'serious' if severity_input == 'yes' else 'minor'
        elif matched_key == 'burn':
            severity_input = input("Is the burn first, second, or third degree? (First/Second/Third): ").strip().lower()
            severity = 'serious' if severity_input in ['second', 'third'] else 'minor'
        elif matched_key == 'chemical burn':
            severity_input = input("Is the chemical non-toxic or corrosive? (Non-toxic/Corrosive): ").strip().lower()
            severity = 'serious' if severity_input == 'corrosive' else 'minor'
        elif matched_key == 'animal bite':
            severity_input = input("Was the animal domestic or wild? (Domestic/Wild): ").strip().lower()
            severity = 'serious' if severity_input == 'wild' else 'minor'
        elif matched_key == 'broken bone':
            severity_input = input("Is the fracture open or closed? (Open/Closed): ").strip().lower()
            severity = 'serious' if severity_input == 'open' else 'minor'
        elif matched_key == 'missing tooth':
            severity_input = input("Is it a baby tooth or permanent tooth? (Baby/Permanent): ").strip().lower()
            severity = 'serious' if severity_input == 'permanent' else 'minor'
        elif matched_key == 'foreign object in nose':
            severity_input = input("Is the object small or large? (Small/Large): ").strip().lower()
            severity = 'serious' if severity_input == 'large' else 'minor'
        elif matched_key == 'bruise':
            severity_input = input("Is it a large bruise that covers a significant area of the body or a small bruise? (Small/Large): ").strip().lower()
            severity = 'serious' if severity_input == 'large' else 'minor'

        # Retrieve and return the appropriate first aid details
        return get_first_aid_details(matched_key, severity)

    # Use GPT for generating a response if the knowledge base doesn't cover it
    system_message = SystemMessage(content="You are a helpful first-aid chatbot.")
    user_message = HumanMessage(content=msg)

    context_memory.append(system_message)
    context_memory.append(user_message)

    ai_response = chat_model.invoke([*context_memory, user_message])
    ai_message = AIMessage(content=ai_response.content)
    context_memory.append(ai_message)

    return ai_response.content

def collect_feedback():
    """Enhanced feedback collection from the user."""
    print("Please provide feedback for this session.")
    rating = input("Rate your experience from 1 to 5 (1 = Poor, 5 = Excellent): ").strip()
    suggestions = input("Do you have any suggestions for improving the chatbot? (Optional): ").strip()

    feedback = {
        "rating": rating,
        "suggestions": suggestions
    }

    # Save feedback to a feedback file
    feedback_file = os.path.join(conversation_dir, 'feedback.json')
    feedback_data = []

    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except json.JSONDecodeError:
            print("Error reading feedback file. Initializing a new one.")

    feedback_data.append(feedback)

    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=4)

    print("Thank you for your feedback!")

def conversation_pdf(conversation, first_aid_info):
    """Generate a PDF report of the conversation and first aid details."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="First Aid Report", ln=True, align="C")
    pdf.ln(10)

    pdf.multi_cell(0, 10, f"Conversation:\n{conversation.encode('latin-1', 'replace').decode('latin-1')}")
    pdf.ln(10)

    # Convert first_aid_info dictionary to a string
    first_aid_info_str = json.dumps(first_aid_info, indent=4)
    pdf.multi_cell(0, 10, f"First Aid Information:\n{first_aid_info_str.encode('latin-1', 'replace').decode('latin-1')}")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    pdf_output_path = os.path.join(conversation_dir, f"conversation_{timestamp}.pdf")
    pdf.output(pdf_output_path)

    return pdf_output_path

def summarize_conversation(conversation):
    """Generate a summary of the conversation using OpenAI GPT-4."""
    system_message = SystemMessage(
        content="You are a helpful assistant. Summarize the following conversation in a few sentences, highlighting the main points.")
    user_message = HumanMessage(content=conversation)

    # Get the summary from GPT-4
    ai_response = chat_model.invoke([system_message, user_message])
    return ai_response.content

class PDF(FPDF):
    def header(self):
        # You can add custom headers here if needed
        pass

    def footer(self):
        # Add footer on each page
        self.set_y(-15)
        self.set_font("DejaVu", size=8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def add_frame(self):
        # Add a frame (rectangle) to every page
        self.set_line_width(0.5)
        self.rect(10, 10, 190, 277)  # Adjust the coordinates as needed

def generate_pdf_report(conversation, first_aid_info, user_details=None, location=None, emergency_contact=None,
                        firstaid_provided=False, pdf_output_path="report.pdf"):
    """
    Generate a PDF report from the chatbot conversation, first aid information, and other details.

    Parameters:
        conversation (str): Full conversation text.
        first_aid_info (dict): First aid information provided.
        user_details (dict): Dictionary with user details (name, age, gender).
        location (str): User's location.
        emergency_contact (str): Emergency contact details.
        firstaid_provided (bool): Whether first aid was provided.
        pdf_output_path (str): File path for saving the PDF report.
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Load fonts (DejaVu fonts need to be added to the directory if not already)
    font_dir = r"C:\Users\Kasthuri\PycharmProjects\CHATBOT01"  # Adjust this path if needed

    # Add the fonts
    pdf.add_font("DejaVu", "", f"{font_dir}/DejaVuSans.ttf", uni=True)  # Regular font
    pdf.add_font("DejaVu", "B", f"{font_dir}/DejaVuSans-Bold.ttf", uni=True)  # Bold font

    # Add the frame to the first page and subsequent pages
    pdf.add_frame()

    # Add the header
    pdf.set_font("DejaVu", style="B", size=14)
    pdf.cell(200, 10, txt="Chatbot Interaction Report", ln=True, align="C")
    pdf.ln(10)

    # Add the Date and Time
    current_time = datetime.now()
    pdf.set_font("DejaVu", size=12)
    pdf.cell(200, 10, txt=f"Date: {current_time.strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(200, 10, txt=f"Time: {current_time.strftime('%H:%M:%S')}", ln=True)
    pdf.ln(10)

    # Add User Details (if available)
    if user_details:
        pdf.cell(200, 10, txt="User Details:", ln=True)
        pdf.cell(200, 10, txt=f"    User Name: {user_details.get('name', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"    User Age: {user_details.get('age', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"    User Gender: {user_details.get('gender', 'N/A')}", ln=True)
        pdf.ln(10)

    # Add the Summary of First Aid
    summary = summarize_conversation(conversation)  # Summarize the conversation
    pdf.cell(200, 10, txt="Summary of First Aid Provided:", ln=True)
    pdf.multi_cell(0, 10, txt=summary)
    pdf.ln(10)

    # Add the Full Conversation
    pdf.cell(200, 10, txt="Conversation:", ln=True)
    pdf.ln(5)
    for message in conversation.split("\n"):
        pdf.multi_cell(0, 10, txt=message)
        pdf.ln(2)

    # Add Location (if available)
    if location:
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Location: {location}", ln=True)

    # Add Emergency Contact (if available)
    if emergency_contact:
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"Emergency Contact: {emergency_contact}", ln=True)

    # Add First Aid Provided Status
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"First Aid Provided: {'Yes' if firstaid_provided else 'No'}", ln=True)

    # Add the frame to the next page if needed
    pdf.add_page()
    pdf.add_frame()

    # Create a unique file name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_output_path = f"chatbot_report_{timestamp}.pdf"

    # Output the PDF
    pdf.output(pdf_output_path)
    print(f"PDF report generated at: {pdf_output_path}")
    return pdf_output_path

def chatbot():
    """Main chatbot loop."""
    conversation = ""
    first_aid_data = {}
    user_details = {}  # Optional: Populate with user-specific information if needed
    location = None
    emergency_contact = None
    firstaid_provided = False  # This can be updated based on your bot's logic

    print("Chatbot is running! Type 'quit' to exit.")

    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            print("Goodbye! Saving your conversation...")

            # Collect feedback before quitting
            collect_feedback()

            # Save the conversation log
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            conversation_file_path = os.path.join(conversation_dir, f"conversation_{timestamp}.txt")
            with open(conversation_file_path, 'w') as file:
                file.write(conversation)

            print(f"Conversation log saved to {conversation_file_path}.")

            print("Session ended. Thank you for your feedback!")
            break

        response = get_response(msg)
        print("Bot: ", response)

        conversation += f"You: {msg}\nBot: {response}\n"

        # Capture first aid responses
        if "serious" in response or "emergency" in response:
            first_aid_data[msg] = response

    first_aid_info = first_aid_data  # Pass the actual dictionary

    report_path = generate_pdf_report(conversation, first_aid_info)
    print(f"Conversation saved: {report_path}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
