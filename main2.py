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

def generate_pdf_report(conversation, first_aid_info):
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


def chatbot():
    """Main chatbot loop."""
    conversation = ""
    first_aid_data = {}
    print("Chatbot is running! Type 'quit' to exit.")

    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            print("Goodbye! Saving your conversation...")

            # Collect feedback before quitting
            feedback = input("Was this session helpful? (Yes/No): ").strip().lower()
            if feedback == "yes":
                # Save feedback-related details
                for question, answer in first_aid_data.items():
                    save_new_solution_to_file(question, answer)

            # Save the conversation log
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            with open(os.path.join(conversation_dir, f"conversation_{timestamp}.txt"), 'w') as file:
                file.write(conversation)

            # Generate PDF report
            generate_pdf_report(conversation, first_aid_data)

            print("Session ended. Thank you for your feedback!")
            break

        response = get_response(msg)
        print("Bot: ", response)

        conversation += f"You: {msg}\nBot: {response}\n"

        if "serious" in response or "emergency" in response:
            first_aid_data[msg] = response


# Run the chatbot
if __name__ == "__main__":
    chatbot()
