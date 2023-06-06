import openai
import numpy as np

direct = 'telemetry.csv'

# Load the telemetry data from the CSV file
telemetry_data = np.array(np.loadtxt(direct, delimiter=','))


def analyze_telemetry():
    # Extract the face and gaze durations from the telemetry data
    face_durations = telemetry_data[:, 0]
    gaze_durations = telemetry_data[:, 1]

    # Perform analysis using GPT-2
    tips = []
    prompt = 'Here is a set of my face and gaze durations in 5 second increments when i was studying,'
    i = 5

    for face_duration, gaze_duration in zip(face_durations, gaze_durations):
        # Generate a prompt based on the face and gaze durations
        prompt = f"Face duration: {face_duration:.2f}s, Gaze duration: {gaze_duration:.2f}s at {i-5} to {i}s\n"

    prompt += "What are some tips to increase productivity and focus? Answer in a fun educational encouraging way"

    # Set up OpenAI API credentials and model
    openai.api_key = ''
    model_name = 'gpt-3.5-turbo'

    # Generate response using GPT-2
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
    )

    # Extract the generated tip from the GPT-2 response
    tip = response.choices[0].message['content'].strip()
    tips.append(tip)

    # Print and return the tips

    return tips
