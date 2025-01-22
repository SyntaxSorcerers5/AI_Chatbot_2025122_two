import openai
import json

# Load knowledge base from the JSON file
with open('knowledge_base.json') as f:
    knowledge_base = json.load(f)

# Set up OpenAI API Key (Replace with your own OpenAI API key)
openai.api_key = "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"  # Replace with your actual OpenAI API key


# Function to query GPT for responses (limited use)
def get_gpt_response(user_input, context="first aid"):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant providing first aid advice. {context}"},
                {"role": "user", "content": user_input}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error with GPT-4 API: {e}"


# Function to classify intent and accident severity
def classify_intent_and_severity(user_input):
    if "cut" in user_input.lower():
        if "deep" in user_input.lower():
            return "deep cut", "high"
        else:
            return "cut", "low"
    elif "burn" in user_input.lower():
        if "serious" in user_input.lower():
            return "serious burn", "high"
        else:
            return "burn", "low"
    elif "choking" in user_input.lower():
        return "choking", "high"
    elif "eye pain" in user_input.lower():
        return "eye pain", "medium"
    else:
        return "unknown", "unknown"


# Function to retrieve treatment and source from the knowledge base
def get_first_aid_info(intent):
    if intent in knowledge_base:
        treatment = knowledge_base[intent].get("treatment", ["No treatment information available."])
        source = knowledge_base[intent].get("source", None)
        return treatment, source
    else:
        return None, None


# Function to provide first aid advice based on severity
def provide_first_aid(intent, severity):
    treatment, source = get_first_aid_info(intent)

    if treatment:
        response = f"Quick First Aid for {intent} (Severity: {severity}):\n{' '.join(treatment)}"
        if source:
            response += f"\n\nSOURCE: {source}"
        else:
            gpt_source = get_gpt_response(f"Provide a credible source for first aid advice about {intent}.")
            response += f"\n\nSOURCE: {gpt_source}"
    else:
        response = f"No information found in the knowledge base for {intent}. Asking GPT for advice..."
        response += f"\n\n{get_gpt_response('Provide first aid advice for ' + intent + ' of severity ' + str(severity) + '.')}"

    return response


# Store first aid information and accident details in a file
def store_accident_and_treatment(accident, treatment, feedback_file='accidents_treatment.json'):
    try:
        with open(feedback_file, 'a') as f:
            json.dump({"accident": accident, "treatment": treatment}, f, indent=4)
            f.write("\n")
    except Exception as e:
        print(f"Error storing accident and treatment: {e}")


# Function to store new problem and solution to a file (if feedback is positive)
def store_new_problem_solution(problem, solution, feedback_file='new_problems_solutions.json'):
    try:
        with open(feedback_file, 'a') as f:
            json.dump({"problem": problem, "solution": solution}, f, indent=4)
            f.write("\n")
    except Exception as e:
        print(f"Error storing new problem and solution: {e}")


# Main chatbot function that integrates the logic for first aid and severity classification
def chatbot_response(user_input, conversation_context, session_memory):
    # Check if the user's input is related to a previously mentioned emergency
    if "continue" in user_input.lower() and session_memory.get("last_intent"):
        intent = session_memory["last_intent"]
        severity = session_memory["last_severity"]
        response = provide_first_aid(intent, severity)
    else:
        intent, severity = classify_intent_and_severity(user_input)

        if intent != "unknown":
            # Update session memory
            session_memory["last_intent"] = intent
            session_memory["last_severity"] = severity

            # Provide first aid advice
            response = provide_first_aid(intent, severity)
        else:
            # Use GPT for follow-up questions to clarify the emergency only when necessary
            response = get_gpt_response(
                f"User's input is unclear: '{user_input}'. Ask a follow-up question to clarify the emergency.")

    # Update the conversation context
    conversation_context += f" User: {user_input}\nBot: {response}\n"
    return response, conversation_context, session_memory


# Interactive chat loop
def start_chat():
    print("Hello! I'm your First Aid Assistant. How can I help you today?")
    conversation_context = ""  # Initialize conversation context
    session_memory = {}  # Memory for current session
    provided_first_aid = False  # Flag to check if first aid was provided
    new_problem = None  # Flag to check if a new problem was encountered
    new_problem_solution = None  # Store GPT solution for new problem

    while True:
        user_input = input("You: ")

        if "bye" in user_input.lower() or "exit" in user_input.lower() or "quit" in user_input.lower():
            print("Goodbye! Stay safe.")

            # Ask for feedback at the end of the conversation
            feedback = input("Was the advice helpful? (yes/no): ").lower()
            if provided_first_aid:
                accident = session_memory.get("last_intent", "Unspecified emergency")
                treatment = "First aid provided based on the user's query."
                if feedback == 'yes':
                    store_accident_and_treatment(accident, treatment)

            # Store new problem and solution if applicable and feedback is good
            if new_problem and feedback == 'yes':
                store_new_problem_solution(new_problem, new_problem_solution)
                print("New problem and solution stored.")

            break

        # Generate the response based on intent classification or GPT fallback
        response, conversation_context, session_memory = chatbot_response(user_input, conversation_context,
                                                                          session_memory)

        # If it's a new problem, store it and its GPT solution
        if "unknown" in response.lower():
            new_problem = user_input
            new_problem_solution = get_gpt_response(f"Provide a solution for {user_input}")
            response += f"\n\nSOLUTION: {new_problem_solution}"

        print("Bot: " + response)

        if "first aid" in response.lower():
            # Mark that first aid was provided
            provided_first_aid = True


# Start the chat loop
if __name__ == "__main__":
    start_chat()
