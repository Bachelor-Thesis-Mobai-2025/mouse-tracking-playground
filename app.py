import os
import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Secret key for session management
app.secret_key = os.urandom(32)

# Make sure these directories exist
os.makedirs('data_new/deceitful', exist_ok=True)
os.makedirs('data_new/truthful', exist_ok=True)

# Instructions for different phases
INSTRUCTIONS = {
    "start": "Thank you for taking the time to answer the following Yes/No questions.\n\n" +
             "During the next 20 questions, we will collect your mouse movements.\n" +
             "As such we ask that you answer these next 10 questions truthfully.\n\n" +
             "NB: The answer choices swap inbetween questions, " +
             "we ask that you take the time to select the right answer.",

    "switch": "For the next 10 questions we ask that you answer the questions deceitfully (Lie).\n\n" +
              "NB: The answer choices swap inbetween questions, " +
              "we ask that you take the time to select the right answer.",

    "end": "You have completed all questions.\n We Thank you for your participation!\n\n" +
           "Please be so kind to sign the \"Data Collection Consent\" form\n" +
           "if you do not object to our use of the mouse tracking data_new collected from you."
}

# Questions (unchanged)
QUESTIONS = [
    # Truthful Yes Answers: Ground truth should be YES
    "Are you currently located in Norway?",
    "Are you currently located in the county (NO: Fylke): Innlandet?",
    "Are you currently located in Gjøvik?",
    "Are you currently located at NTNU Gjøvik?",
    "Are you currently a Student at NTNU?",
    # Truthful No Answers: Ground truth should be NO
    "Are you currently located in the USA?",
    "Are you currently located in the US state of California?",
    "Are you currently located in Berkeley?",
    "Are you currently located at University of California, Berkeley?",
    "Are you currently a student at University of California, Berkeley?",
    # Skipped to show instructions for deception
    "",
    # Deceitful Yes Answers: Ground truth should be NO
    "Are you currently located in Australia?",
    "Are you currently located in the AU state of Victoria?",
    "Are you currently located in Melbourne?",
    "Are you currently located at the Royal Melbourne Institute of Technology?",
    "Are you currently employed as a Professor at the Royal Melbourne Institute of Technology?",
    # Deceitful No Answers: Ground truth should be YES
    "Are you currently located in Norway?",
    "Are you currently located in the county (NO: Fylke): Innlandet?",
    "Are you currently located in Gjøvik?",
    "Are you currently located at NTNU Gjøvik?",
    "Are you currently a Student at NTNU?",
]

# Define the ground truth for each question (1 for yes, 0 for no)
GROUND_TRUTH = [
    # Truthful Yes (first 5)
    1, 1, 1, 1, 1,
    # Truthful No (next 5)
    0, 0, 0, 0, 0,
    # Deceptive Yes (next 5)
    0, 0, 0, 0, 0,
    # Deceptive No (last 5)
    1, 1, 1, 1, 1
]


@app.route('/')
def index():
    # Reset session data_new when starting a new session
    session['question_count'] = 0
    session['asked_truthful_indices'] = []
    session['asked_deceptive_indices'] = []
    session['phase'] = 'truthful'
    return render_template('index.html')


@app.route('/log_data', methods=['POST'])
def log_data():
    # Get the JSON data_new from the request
    trajectory_data = request.json

    # Get the last asked question index from session
    last_question_index = session.get('last_question_index', 0)

    # Get user's answer (1 for yes, 0 for no)
    user_answer = trajectory_data.get('answer', None)
    if user_answer is None:
        return jsonify({"status": "error", "message": "No answer provided"})

    # Adjust index for deceitful phase
    if session.get('phase') == 'deceitful':
        last_question_index -= 1

    # Get ground truth for this question
    ground_truth = GROUND_TRUTH[last_question_index]

    # Determine if the answer was truthful
    is_truthful = user_answer == ground_truth

    # Determine answer direction (yes/no)
    answer_suffix = "yes" if user_answer == 1 else "no"

    # Select folder based on truthfulness
    subfolder = 'truthful' if is_truthful else 'deceitful'

    # Add truthfulness label to the data_new
    trajectory_data['label'] = 'truthful' if is_truthful else 'deceitful'

    # Remove any potentially biasing information
    if 'question' in trajectory_data:
        del trajectory_data['question']
    if 'answer' in trajectory_data:
        del trajectory_data['answer']
    if 'answer' in trajectory_data:
        del trajectory_data['answer']

    # Create filename with timestamp and answer suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data_new/{subfolder}/tracking_{timestamp}_{answer_suffix}.json'

    # Save to JSON file
    with open(filename, 'w') as jsonfile:
        json.dump(trajectory_data, jsonfile, indent=2)

    return jsonify({"status": "success"})


def get_random_question_index(phase):
    """Get a random question index that hasn't been asked yet for the current phase."""
    # Define the range of indices based on the phase
    if phase == 'truthful':
        # For truthful phase, use questions 0-9 (first 10 questions)
        available_indices = list(range(10))
        already_asked = session.get('asked_truthful_indices', [])
    else:  # 'deceitful' phase
        # For deceitful phase, use questions 11-21 (skip the empty question at index 10)
        available_indices = list(range(11, 21))
        already_asked = session.get('asked_deceptive_indices', [])

    # Filter out questions that have already been asked
    remaining_indices = [idx for idx in available_indices if idx not in already_asked]

    # If we've asked all questions in this phase, return None
    if not remaining_indices:
        return None

    # Select a random question from the remaining ones
    return random.choice(remaining_indices)


@app.route('/get_question')
def get_question():
    # Get current question count
    question_count = session.get('question_count', 0)
    phase = session.get('phase', 'truthful')

    # Check if we need to show instructions
    if question_count == 0:
        # Starting instructions - first time
        session['question_count'] = 1
        session['phase'] = 'truthful'
        session['asked_truthful_indices'] = []
        session['asked_deceptive_indices'] = []

        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["start"]
        })
    elif question_count == 11:
        # Switch to deceitful mode instructions
        session['phase'] = 'deceitful'
        session['question_count'] = 12

        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["switch"]
        })
    elif question_count >= 22:
        # End of experiment
        return jsonify({
            "isInstruction": True,
            "instruction": INSTRUCTIONS["end"],
            "complete": True
        })

    # Get a random question index for the current phase
    question_index = get_random_question_index(phase)

    # Store the last question index for logging purposes
    session['last_question_index'] = question_index

    # Add this question to the list of asked questions
    if phase == 'truthful':
        asked_indices = session.get('asked_truthful_indices', [])
        asked_indices.append(question_index)
        session['asked_truthful_indices'] = asked_indices
    else:  # 'deceitful' phase
        asked_indices = session.get('asked_deceptive_indices', [])
        asked_indices.append(question_index)
        session['asked_deceptive_indices'] = asked_indices

    # Increment question count
    session['question_count'] = question_count + 1

    # Return the selected question
    return jsonify({
        "isInstruction": False,
        "question": QUESTIONS[question_index]
    })


if __name__ == '__main__':
    app.run(debug=True)
