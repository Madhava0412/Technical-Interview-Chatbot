import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import (
    create_technical_question_prompt,
    create_tech_stack_evaluation_prompt,
    create_error_recovery_prompt,
    create_clarification_prompt,
    create_follow_up_prompt,
    create_interview_conclusion_prompt
)
from output import generate_response

# Set page config
st.set_page_config(
    page_title="Technical Interview Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'interviewer' not in st.session_state:
    st.session_state.interviewer = None
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Tech stack dictionary organized by categories
TECH_STACK_CATEGORIES = {
    "Programming Languages": {
        "Python": "Python programming language",
        "JavaScript": "JavaScript programming language",
        "Java": "Java programming language",
        "C++": "C++ programming language",
        "C#": "C# programming language",
        "Go": "Go programming language",
        "Rust": "Rust programming language",
        "Ruby": "Ruby programming language",
        "PHP": "PHP programming language",
        "TypeScript": "TypeScript programming language",
        "Swift": "Swift programming language",
        "Kotlin": "Kotlin programming language"
    },
    "Web Frameworks": {
        "Django": "Django web framework (Python)",
        "Flask": "Flask web framework (Python)",
        "FastAPI": "FastAPI web framework (Python)",
        "React": "React frontend library (JavaScript)",
        "Angular": "Angular frontend framework (TypeScript)",
        "Vue.js": "Vue.js frontend framework (JavaScript)",
        "Spring Boot": "Spring Boot framework (Java)",
        "Express.js": "Express.js backend framework (Node.js)",
        "Laravel": "Laravel framework (PHP)",
        "Ruby on Rails": "Ruby on Rails framework (Ruby)",
        "ASP.NET Core": "ASP.NET Core framework (C#)"
    },
    "Databases": {
        "PostgreSQL": "PostgreSQL relational database",
        "MySQL": "MySQL relational database",
        "MongoDB": "MongoDB NoSQL database",
        "Redis": "Redis in-memory data store",
        "Elasticsearch": "Elasticsearch search engine",
        "Oracle": "Oracle database",
        "MS SQL Server": "Microsoft SQL Server database",
        "Cassandra": "Apache Cassandra distributed database",
        "Neo4j": "Neo4j graph database"
    },
    "DevOps & Cloud": {
        "Docker": "Docker containerization platform",
        "Kubernetes": "Kubernetes container orchestration",
        "AWS": "Amazon Web Services cloud platform",
        "Azure": "Microsoft Azure cloud platform",
        "GCP": "Google Cloud Platform",
        "Jenkins": "Jenkins automation server",
        "GitLab CI/CD": "GitLab CI/CD pipeline",
        "GitHub Actions": "GitHub Actions automation",
        "Terraform": "Terraform infrastructure as code",
        "Ansible": "Ansible automation tool"
    },
    "Frontend Technologies": {
        "HTML5": "HTML5 markup language",
        "CSS3": "CSS3 styling language",
        "SASS/SCSS": "SASS/SCSS CSS preprocessor",
        "Tailwind CSS": "Tailwind CSS utility framework",
        "Bootstrap": "Bootstrap CSS framework",
        "Material UI": "Material UI component library",
        "Redux": "Redux state management",
        "GraphQL": "GraphQL query language",
        "REST API": "RESTful API architecture"
    },
    "Tools & Others": {
        "Git": "Git version control",
        "Linux": "Linux operating system",
        "Nginx": "Nginx web server",
        "RabbitMQ": "RabbitMQ message broker",
        "Kafka": "Apache Kafka streaming platform",
        "Celery": "Celery task queue",
        "Prometheus": "Prometheus monitoring",
        "Grafana": "Grafana analytics platform",
        "ELK Stack": "Elasticsearch, Logstash, Kibana stack"
    }
}

# Experience level definitions
EXPERIENCE_LEVELS = [
    "0-1 years (Entry Level)",
    "1-3 years (Junior)",
    "3-5 years (Mid-Level)",
    "5-8 years (Senior)",
    "8-12 years (Lead/Principal)",
    "12+ years (Architect/Expert)"
]

# Model configuration
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model with specific device configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device if device == "cuda" else None  # Let the model choose device map for GPU
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model is on the correct device
    if device == "cuda":
        model = model.cuda()
    
    # Display device information
    if device == "cuda":
        st.success(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Model loaded on CPU")
    
    return model, tokenizer, device

class TechnicalInterviewer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conversation_history = []
        self.candidate_skill_level = None
        self.tech_stack = None
        self.years_of_experience = None
        
    def set_candidate_context(self, skill_level, tech_stack, years_of_experience):
        self.candidate_skill_level = skill_level
        self.tech_stack = tech_stack
        self.years_of_experience = years_of_experience
        
    def start_interview(self):
        prompt = create_technical_question_prompt(
            skill_level=f"{self.candidate_skill_level} with {self.years_of_experience}",
            tech_stack=self.tech_stack
        )
        response = self.generate_response(prompt)
        self.conversation_history.append({"role": "interviewer", "content": response})
        return response
    
    def ask_follow_up(self, candidate_response):
        # Record candidate's response first
        self.conversation_history.append({"role": "candidate", "content": candidate_response})
        
        # Then generate follow-up question based on candidate's response
        prompt = create_follow_up_prompt(
            conversation_history=self.conversation_history,
            candidate_response=candidate_response,
            skill_level=f"{self.candidate_skill_level} with {self.years_of_experience}"
        )
        response = self.generate_response(prompt)
        
        # Add interviewer's follow-up question to conversation history
        self.conversation_history.append({"role": "interviewer", "content": response})
        return response
    
    def request_clarification(self, unclear_response):
        # Record candidate's unclear response first
        self.conversation_history.append({"role": "candidate", "content": unclear_response})
        
        # Generate clarification request
        prompt = create_clarification_prompt(
            unclear_response=unclear_response,
            topic=self.tech_stack
        )
        response = self.generate_response(prompt)
        
        # Add interviewer's clarification request to conversation history
        self.conversation_history.append({"role": "interviewer", "content": response})
        return response
    
    def handle_error(self, error_description):
        # Record candidate's error description first
        self.conversation_history.append({"role": "candidate", "content": error_description})
        
        # Generate error recovery response
        prompt = create_error_recovery_prompt(
            error_description=error_description,
            tech_stack=self.tech_stack
        )
        response = self.generate_response(prompt)
        
        # Add interviewer's error recovery response to conversation history
        self.conversation_history.append({"role": "interviewer", "content": response})
        return response
    
    def conclude_interview(self):
        prompt = create_interview_conclusion_prompt(
            conversation_history=self.conversation_history,
            skill_level=f"{self.candidate_skill_level} with {self.years_of_experience}",
            tech_stack=self.tech_stack
        )
        response = self.generate_response(prompt)
        self.conversation_history.append({"role": "interviewer", "content": response})
        return response
    
    def generate_response(self, prompt):
        return generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
        )
    
    def evaluate_tech_stack_knowledge(self):
        prompt = create_tech_stack_evaluation_prompt(
            conversation_history=self.conversation_history,
            tech_stack=self.tech_stack
        )
        response = self.generate_response(prompt)
        return response
    
# Main app
def main():
    st.title("🤖 Technical Interview Chatbot")
    st.markdown("An AI-powered technical interviewer to help prepare for coding interviews")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not st.session_state.model_loaded:
            if st.button("Load Model"):
                with st.spinner("Loading model... This may take a few minutes."):
                    model, tokenizer, device = load_model()
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.session_state.device = device
                    st.success("Model loaded successfully!")
        
        if st.session_state.model_loaded:
            st.subheader("Interview Settings")
            
            # Years of experience selection
            years_of_experience = st.selectbox(
                "Years of Experience",
                EXPERIENCE_LEVELS,
                index=2
            )
            
            # Extract skill level from experience
            if "Entry Level" in years_of_experience:
                skill_level = "entry"
            elif "Junior" in years_of_experience:
                skill_level = "junior"
            elif "Mid-Level" in years_of_experience:
                skill_level = "intermediate"
            elif "Senior" in years_of_experience:
                skill_level = "senior"
            elif "Lead/Principal" in years_of_experience:
                skill_level = "lead"
            else:
                skill_level = "expert"
            
            st.subheader("Tech Stack Selection")
            
            selected_tech = {}
            
            # Create expandable sections for each category
            for category, technologies in TECH_STACK_CATEGORIES.items():
                with st.expander(f"📚 {category}", expanded=False):
                    selected_items = st.multiselect(
                        f"Select {category}",
                        options=list(technologies.keys()),
                        key=f"select_{category.lower().replace(' ', '_')}"
                    )
                    selected_tech[category] = selected_items
            
            # Flatten selected technologies into a single list
            tech_stack = []
            for category, items in selected_tech.items():
                tech_stack.extend(items)
            
            if len(tech_stack) > 0:
                st.success(f"Selected technologies: {', '.join(tech_stack)}")
            else:
                st.warning("Please select at least one technology")
            
            if st.button("Start Interview") and not st.session_state.interview_started and len(tech_stack) > 0:
                st.session_state.interviewer = TechnicalInterviewer(st.session_state.model, st.session_state.tokenizer, st.session_state.device)
                st.session_state.interviewer.set_candidate_context(skill_level, tech_stack, years_of_experience)
                
                # Start the interview
                first_question = st.session_state.interviewer.start_interview()
                st.session_state.conversation_history.extend(st.session_state.interviewer.conversation_history)
                st.session_state.interview_started = True
            
            if st.session_state.interview_started:
                st.warning("Interview in progress")
                if st.button("End Interview"):
                    conclusion = st.session_state.interviewer.conclude_interview()
                    st.session_state.conversation_history.append({"role": "interviewer", "content": conclusion})
                    evaluation = st.session_state.interviewer.evaluate_tech_stack_knowledge()
                    st.session_state.conversation_history.append({"role": "evaluation", "content": evaluation})
                    st.session_state.interview_started = False
                    st.success("Interview concluded!")
    
    # Main chat interface
    st.subheader("💬 Interview Chat")
    
    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation_history:
            if message["role"] == "interviewer":
                with st.chat_message("assistant"):
                    st.write(message["content"])
            elif message["role"] == "candidate":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "evaluation":
                with st.expander("📊 Tech Stack Evaluation"):
                    st.write(message["content"])
    
    # Input area for candidate responses
    if st.session_state.interview_started:
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input("Your response:", key="user_input")
        
        with col2:
            if st.button("Send"):
                if user_input:
                    # Get follow-up question
                    with st.spinner("Generating response..."):
                        response = st.session_state.interviewer.ask_follow_up(user_input)
                        st.session_state.conversation_history = st.session_state.interviewer.conversation_history.copy()
                    st.rerun()
        
        # Special actions
        st.markdown("---")
        st.subheader("Special Actions")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            clarification_input = st.text_input("Unclear response:")
            if st.button("Request Clarification"):
                if clarification_input:
                    with st.spinner("Generating clarification request..."):
                        response = st.session_state.interviewer.request_clarification(clarification_input)
                        st.session_state.conversation_history = st.session_state.interviewer.conversation_history.copy()
                    st.rerun()
        
        with col4:
            error_input = st.text_input("Error description:")
            if st.button("Report Error"):
                if error_input:
                    with st.spinner("Generating error recovery response..."):
                        response = st.session_state.interviewer.handle_error(error_input)
                        st.session_state.conversation_history = st.session_state.interviewer.conversation_history.copy()
                    st.rerun()
        
        with col5:
            if st.button("Evaluate Tech Stack"):
                with st.spinner("Evaluating tech stack knowledge..."):
                    evaluation = st.session_state.interviewer.evaluate_tech_stack_knowledge()
                    st.session_state.conversation_history.append({"role": "evaluation", "content": evaluation})
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This technical interview chatbot uses the Microsoft Phi-3 model to simulate a real technical interview experience.")
    st.markdown("It helps candidates prepare for technical interviews by asking relevant questions and providing feedback.")

if __name__ == "__main__":
    main()