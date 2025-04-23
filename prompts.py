import re
def create_technical_question_prompt(skill_level, tech_stack):
    """
    Create a prompt template for generating technical questions.
    
    Args:
        skill_level (str): Skill level with years of experience (e.g., "intermediate with 3-5 years")
        tech_stack (list): List of technologies the candidate knows
        
    Returns:
        str: Formatted prompt template ready for model input
    """
    # Extract years from skill level string
    if "Entry Level" in skill_level or "0-1 years" in skill_level:
        experience_years = 1
        experience_level = "entry"
        complexity = "fundamental"
        focus_areas = "basic concepts, syntax basics, simple implementations, and common patterns"
    elif "Junior" in skill_level or "1-3 years" in skill_level:
        experience_years = 2
        experience_level = "junior"
        complexity = "beginner to intermediate"
        focus_areas = "fundamental concepts, basic syntax, simple debugging, and common use cases"
    elif "Mid-Level" in skill_level or "3-5 years" in skill_level:
        experience_years = 4
        experience_level = "intermediate"
        complexity = "intermediate"
        focus_areas = "design patterns, best practices, problem-solving, and performance optimization"
    elif "Senior" in skill_level or "5-8 years" in skill_level:
        experience_years = 6
        experience_level = "senior"
        complexity = "advanced"
        focus_areas = "system design, scalability, architectural decisions, and complex problem-solving"
    elif "Lead" in skill_level or "8-12 years" in skill_level:
        experience_years = 10
        experience_level = "lead"
        complexity = "advanced to expert"
        focus_areas = "architectural decisions, team leadership, complex problem-solving, and technology strategy"
    else:  # Architect/Expert
        experience_years = 15
        experience_level = "expert"
        complexity = "expert"
        focus_areas = "enterprise architecture, strategic technology decisions, industry trends, and organizational tech leadership"
    
    # Select a primary technology from the tech stack
    primary_tech = tech_stack[0] if tech_stack else "general software development"
    
    prompt = f"""You are an expert technical interviewer conducting an interview with a {experience_level} developer. Generate ONE insightful technical interview question for a candidate with approximately {experience_years} years of experience.

Primary Technology: {primary_tech}
Tech Stack Context: {', '.join(tech_stack)}
Experience Level: {experience_level}
Question Complexity: {complexity}

Focus on:
- {focus_areas}

Requirements:
1. Question should be practical and scenario-based
2. Relate to real-world problems they might face
3. Allow for demonstration of their experience level
4. Consider their full tech stack when applicable
5. Be open-ended enough to encourage detailed discussion

Write the question as if you're directly asking the candidate. Be conversational but professional.

Generate your question now:"""
    
    return prompt


def create_follow_up_prompt(conversation_history, candidate_response, skill_level):
    """
    Create a prompt template for generating follow-up responses based on candidate's answer.
    
    Args:
        conversation_history (list): List of previous messages in the conversation
        candidate_response (str): Candidate's latest answer
        skill_level (str): Skill level with years of experience
        
    Returns:
        str: Formatted prompt for generating follow-up
    """
    # Extract the last question from conversation history
    last_question = None
    for message in reversed(conversation_history):
        if message["role"] == "interviewer":
            last_question = message["content"]
            break
    
    # Extract tech stack from conversation
    tech_stack = []
    for message in conversation_history:
        if message["role"] == "interviewer" and "Tech Stack Context:" in message.get("content", ""):
            tech_context = message["content"].split("Tech Stack Context:")[1].split("\n")[0].strip()
            tech_stack = [tech.strip() for tech in tech_context.split(',')]
            break
    
    prompt = f"""You are an expert technical interviewer reviewing a candidate's response. Respond appropriately based on the quality and content of their answer.

Candidate Skill Level: {skill_level}
Previous Question: {last_question}
Candidate's Response: {candidate_response}
Tech Stack: {', '.join(tech_stack) if tech_stack else 'Not specified'}

Analyze the candidate's response and choose the most appropriate response type:

If the response is NONSENSICAL or COMPLETELY IRRELEVANT (e.g., random characters, off-topic, or unintelligible):
- Acknowledge politely that you didn't understand their response
- Ask them if they could clarify or rephrase
- Suggest they might want to address the original question

If the response demonstrates MISUNDERSTANDING:
- Clarify the original question
- Provide a hint to guide them in the right direction
- Ask a simpler version of the question

If the response is PARTIALLY CORRECT:
- Acknowledge the correct parts
- Probe deeper on areas that need elaboration
- Ask for specific examples or scenarios

If the response is TECHNICALLY SOUND:
- Acknowledge their good answer
- Ask a follow-up question that builds on their response
- Challenge them with a related but more complex scenario

Keep your response conversational, encouraging, and focused on assessing their technical knowledge. Respond directly as if speaking to the candidate.

Your response:"""
    
    return prompt


def create_tech_stack_evaluation_prompt(conversation_history, tech_stack):
    """
    Create a prompt template for evaluating candidate's performance on tech stack questions.
    
    Args:
        conversation_history (list): Full conversation history
        tech_stack (list): Technologies the candidate claims to know
        
    Returns:
        str: Formatted prompt for tech stack evaluation
    """
    # Extract candidate responses for analysis
    candidate_responses = []
    for message in conversation_history:
        if message["role"] == "candidate":
            candidate_responses.append(message["content"])
    
    combined_responses = "\n\n".join(candidate_responses)
    
    prompt = f"""You are an expert technical interviewer analyzing a candidate's performance across their stated tech stack.

Tech Stack: {', '.join(tech_stack)}

Review these candidate responses from the interview:
{combined_responses}

Based on their responses, provide:
1. Assessment of their depth of knowledge for each technology discussed
2. Identification of any gaps between claimed and demonstrated knowledge
3. Specific areas where they showed strength or weakness
4. Overall rating of their technical proficiency
5. Recommendations for improvement

Format your response as:
Technical Assessment: [Brief overall assessment]
Strengths: [Key technical strengths observed]
Areas for Improvement: [Specific gaps or weaknesses]
Recommendations: [Actionable improvement suggestions]
Overall Rating: [Rate from 1-5 stars with justification]"""
    
    return prompt


def create_interview_conclusion_prompt(conversation_history, skill_level, tech_stack):
    """
    Create a prompt template for generating a conclusion to the interview.
    
    Args:
        conversation_history (list): Full conversation history
        skill_level (str): Skill level with years of experience
        tech_stack (list): Technologies discussed
        
    Returns:
        str: Formatted prompt for generating conclusion
    """
    # Extract a few candidate responses for context
    candidate_responses = []
    for message in conversation_history:
        if message["role"] == "candidate":
            candidate_responses.append(message["content"])
    
    # Get the last 3 responses for context
    recent_responses = candidate_responses[-3:] if len(candidate_responses) >= 3 else candidate_responses
    
    prompt = f"""You are an expert technical interviewer concluding an interview session. Generate a professional and encouraging conclusion.

Interview Context:
- Candidate Experience Level: {skill_level}
- Technologies Covered: {', '.join(tech_stack)}

Some of the candidate's recent responses:
{"\n\n".join(recent_responses)}

Create a conclusion that:
1. Thanks the candidate for their time and thoughtful responses
2. Highlights 1-2 particularly strong points from their interview
3. Mentions any specific areas that showed promise or expertise
4. Explains the typical next steps in the hiring process
5. Provides an estimated timeline for feedback
6. Ends on an encouraging and professional note

Keep the tone friendly yet professional, and limit to 4-5 sentences."""
    
    return prompt


def create_clarification_prompt(unclear_response, topic):
    """
    Create a prompt template for generating clarification requests when input is unclear.
    
    Args:
        unclear_response (str): The unclear response from candidate
        topic (list): Current technical topics being discussed
        
    Returns:
        str: Formatted prompt for generating clarification
    """
    prompt = f"""You are an expert technical interviewer who has received an unclear response from a candidate. Generate a professional clarification request.

Topic Being Discussed: {', '.join(topic) if isinstance(topic, list) else topic}
Unclear Response: "{unclear_response}"

Create a response that:
1. Acknowledges their attempt to answer without being condescending
2. Politely indicates what part was unclear or needs elaboration
3. Asks specific follow-up questions to guide them
4. Suggests an aspect they might want to focus on
5. Maintains an encouraging and supportive tone

Keep your response conversational and helpful. Respond directly as if speaking to the candidate."""
    
    return prompt


def create_error_recovery_prompt(error_description, tech_stack):
    """
    Create a prompt template for recovering from errors or misconceptions in answers.
    
    Args:
        error_description (str): Description of the error or misconception
        tech_stack (list): Technologies being discussed
        
    Returns:
        str: Formatted prompt for error recovery
    """
    prompt = f"""You are an expert technical interviewer who needs to address a misconception or error in a candidate's response. Generate a supportive correction.

Error or Misconception: "{error_description}"
Technical Context: {', '.join(tech_stack)}

Create a response that:
1. Acknowledges their effort without being condescending
2. Gently corrects the misconception with proper technical information
3. Provides a simple example or analogy to clarify
4. Asks a follow-up question to check understanding
5. Maintains the candidate's confidence

Your response should be educational but not lengthy. Respond directly as if speaking to the candidate."""
    
    return prompt


def detect_nonsensical_input(response):
    """
    Detect if a candidate response is nonsensical or completely irrelevant.
    
    Args:
        response (str): Candidate's response to analyze
        
    Returns:
        bool: True if the response appears nonsensical, False otherwise
    """
    # Check if response is too short (just a few characters)
    if len(response.strip()) < 5:
        return True
        
    # Check for random character sequences (e.g., "asdfghjkl")
    if re.match(r'^[a-z]{8,}$', response.strip().lower()):
        return True
        
    # Check for responses that are just punctuation or special characters
    if re.match(r'^[^a-zA-Z0-9\s]*$', response.strip()):
        return True
        
    # Check for extremely repetitive content
    chars = response.strip().lower()
    if len(set(chars)) < 3 and len(chars) > 5:  # e.g., "aaaaaaaa" or "hahahaha"
        return True
        
    return False