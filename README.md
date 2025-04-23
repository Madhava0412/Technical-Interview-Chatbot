# Technical Interview Chatbot

An AI-powered technical interviewer to help prepare for coding interviews. This application simulates a technical interview experience using Microsoft's Phi-3 model to ask relevant questions based on your selected technology stack and experience level.


## Features

- 🤖 Dynamic interview questions based on your tech stack and experience level
- 💬 Natural conversation flow with relevant follow-up questions
- 📊 Evaluation of your technical knowledge across different technologies
- 🛠️ Support for various programming languages, frameworks, and tools
- 🔄 Special actions to request clarification or report errors
- 📝 Interview conclusion with personalized feedback

## Tech Stack

- **Frontend**: Streamlit
- **AI Model**: Microsoft Phi-3-mini-4k-instruct
- **Libraries**: PyTorch, Transformers, Streamlit

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Madhava0412/Technical-Interview-Chatbot.git
   cd technical-interview-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Configure your interview:
   - Click "Load Model" (this may take a few minutes)
   - Select your years of experience
   - Choose technologies from the tech stack categories
   - Click "Start Interview"

4. Interact with the chatbot:
   - Respond to questions in the chat interface
   - Use special actions if needed (request clarification, report errors)
   - Get feedback on your tech stack knowledge

5. Conclude the interview when you're done to receive an overall evaluation

## How It Works

The application uses a language model (Microsoft Phi-3) to simulate a technical interviewer. It tailors questions based on:

1. **Experience Level**: Questions are adjusted for entry-level to expert developers
2. **Tech Stack**: Questions focus on the technologies you've selected
3. **Response Analysis**: Follow-up questions are based on your previous answers

The system uses different prompt templates to generate:
- Initial technical questions
- Follow-up questions
- Clarification requests
- Error recovery responses
- Interview conclusions
- Tech stack evaluations

## Configuration Options

### Experience Levels
- 0-1 years (Entry Level)
- 1-3 years (Junior)
- 3-5 years (Mid-Level)
- 5-8 years (Senior)
- 8-12 years (Lead/Principal)
- 12+ years (Architect/Expert)

### Tech Stack Categories
- Programming Languages
- Web Frameworks
- Databases
- DevOps & Cloud
- Frontend Technologies
- Tools & Others

## File Structure

- `app.py`: Main application file with Streamlit UI
- `prompts.py`: Contains prompt templates for different interview scenarios
- `output.py`: Handles generating responses from the language model
- `requirements.txt`: List of required Python packages

## Extending the Application

### Adding New Technologies
Edit the `TECH_STACK_CATEGORIES` dictionary in `app.py` to add new technologies:

```python
TECH_STACK_CATEGORIES = {
    "Category Name": {
        "Tech Name": "Tech Description",
        # Add more technologies here
    },
    # Add more categories here
}
```

### Customizing Prompts
Modify the prompt templates in `prompts.py` to change how the chatbot interacts:

- `create_technical_question_prompt`: Initial questions
- `create_follow_up_prompt`: Follow-up questions
- `create_clarification_prompt`: Clarification requests
- `create_error_recovery_prompt`: Error recovery responses
- `create_interview_conclusion_prompt`: Interview conclusions
- `create_tech_stack_evaluation_prompt`: Tech stack evaluations

## Performance Notes

- The application works best with GPU acceleration but will also run on CPU
- Initial model loading may take 2-5 minutes depending on your hardware
- Response generation typically takes 1-5 seconds

## Troubleshooting

- **Out of memory errors**: Reduce batch size or use a smaller model
- **Slow responses**: Enable GPU acceleration if available
- **Model loading fails**: Ensure you have enough disk space and memory

## License

[MIT License](LICENSE)

## Acknowledgements

- Microsoft for the Phi-3 model
- Hugging Face for the Transformers library
- Streamlit for the web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
