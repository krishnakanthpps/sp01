import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Load Azure OpenAI settings
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Azure OpenAI API URL
api_url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"

def analyze_website_requirements(prompt):
    """
    Analyze a website prompt and produce a comprehensive breakdown of requirements.
    Returns structured JSON data with categorized tasks and requirements.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    # Enhanced system prompt for more comprehensive requirements analysis
    system_prompt = """
    You are a senior website requirements analyst and project planner. Your job is to:
    
    1. Analyze user requests for website creation
    2. Identify ALL necessary components and requirements
    3. Categorize requirements into clear sections
    4. Identify any potential missing information
    5. Create a comprehensive implementation plan
    
    For each website request, produce a structured JSON output with the following format:
    
    {
        "website_name": "Name of the website",
        "primary_purpose": "Main function/purpose of the website",
        "target_audience": "Primary users of the website",
        "sections": {
            "content": [List of content requirements],
            "design": [List of design requirements],
            "functionality": [List of functional requirements],
            "technical": [List of technical requirements]
        },
        "key_pages": [List of pages that should be created],
        "missing_information": [List of critical details that are missing from the request],
        "implementation_tasks": [List of specific tasks to implement the website],
        "completion_checklist": [Items to verify before considering the website complete]
    }
    
    Be thorough and comprehensive. Leave no requirement unspecified. For any vague request, provide reasonable defaults based on industry standards while noting the ambiguity.
    """

    # Define the request payload
    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Please analyze this website request and provide comprehensive requirements: {prompt}"
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        requirements_json = response.json()["choices"][0]["message"]["content"]
        requirements = json.loads(requirements_json)
        return requirements
    else:
        raise Exception(f"Error calling Azure OpenAI API: {response.text}")

def check_requirements_completeness(requirements):
    """
    Validate the completeness of the requirements and identify any critical gaps.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    system_prompt = """
    You are a website requirements validator. Your job is to examine a set of website requirements and:
    
    1. Identify any critical gaps or missing elements
    2. Assess the completeness of each section
    3. Highlight areas that need more detail
    4. Suggest additional requirements that might have been overlooked
    
    Provide your assessment as a JSON object with the following structure:
    
    {
        "completeness_score": 0-100,
        "critical_gaps": [List of critical requirements that are missing],
        "section_scores": {
            "content": 0-100,
            "design": 0-100,
            "functionality": 0-100,
            "technical": 0-100
        },
        "improvement_suggestions": [Specific suggestions to improve requirements],
        "additional_requirements": [Additional requirements that should be considered]
    }
    
    Be thorough but realistic in your assessment.
    """

    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Please assess the completeness of these website requirements: {json.dumps(requirements)}"
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        assessment_json = response.json()["choices"][0]["message"]["content"]
        assessment = json.loads(assessment_json)
        return assessment
    else:
        raise Exception(f"Error calling Azure OpenAI API: {response.text}")

def generate_follow_up_questions(requirements, assessment):
    """
    Generate follow-up questions based on missing information and completeness assessment.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    system_prompt = """
    You are a website requirements consultant who specializes in identifying information gaps and asking the right questions to create comprehensive website specifications.
    
    Based on the initial requirements and completeness assessment, generate a list of specific, targeted questions to fill in the gaps.
    
    Return your questions as a JSON array where each question includes:
    1. The question text
    2. The category it belongs to
    3. Why this information is important
    
    Format your response as:
    {
        "follow_up_questions": [
            {
                "question": "Question text here?",
                "category": "design|content|functionality|technical",
                "importance": "Why this information matters"
            }
        ]
    }
    
    Limit your questions to the 5-7 most important gaps that need to be filled.
    """

    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Here are the initial website requirements:\n{json.dumps(requirements)}\n\nAnd here is the completeness assessment:\n{json.dumps(assessment)}\n\nWhat follow-up questions should I ask to improve the requirements?"
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        questions_json = response.json()["choices"][0]["message"]["content"]
        questions = json.loads(questions_json)
        return questions
    else:
        raise Exception(f"Error calling Azure OpenAI API: {response.text}")

def display_requirements(requirements):
    """Display the requirements in a formatted way"""
    print("\n" + "="*80)
    print(f"📝 WEBSITE REQUIREMENTS: {requirements['website_name']}")
    print("="*80)
    
    print(f"\n🎯 PRIMARY PURPOSE: {requirements['primary_purpose']}")
    print(f"👥 TARGET AUDIENCE: {requirements['target_audience']}")
    
    print("\n📄 KEY PAGES:")
    for i, page in enumerate(requirements['key_pages'], 1):
        print(f"  {i}. {page}")
    
    print("\n🧩 REQUIREMENTS BY CATEGORY:")
    for category, items in requirements['sections'].items():
        print(f"\n  📌 {category.upper()}:")
        for i, item in enumerate(items, 1):
            print(f"    {i}. {item}")
    
    print("\n⚠️ MISSING INFORMATION:")
    for i, item in enumerate(requirements['missing_information'], 1):
        print(f"  {i}. {item}")
    
    print("\n🔧 IMPLEMENTATION TASKS:")
    for i, task in enumerate(requirements['implementation_tasks'], 1):
        print(f"  {i}. {task}")

def display_assessment(assessment):
    """Display the completeness assessment in a formatted way"""
    print("\n" + "="*80)
    print(f"🔍 COMPLETENESS ASSESSMENT - Score: {assessment['completeness_score']}/100")
    print("="*80)
    
    print("\n⚠️ CRITICAL GAPS:")
    for i, gap in enumerate(assessment['critical_gaps'], 1):
        print(f"  {i}. {gap}")
    
    print("\n📊 SECTION SCORES:")
    for section, score in assessment['section_scores'].items():
        print(f"  {section.capitalize()}: {score}/100")
    
    print("\n💡 IMPROVEMENT SUGGESTIONS:")
    for i, suggestion in enumerate(assessment['improvement_suggestions'], 1):
        print(f"  {i}. {suggestion}")

def display_follow_up_questions(questions):
    """Display follow-up questions in a formatted way"""
    print("\n" + "="*80)
    print("❓ FOLLOW-UP QUESTIONS")
    print("="*80 + "\n")
    
    for i, q in enumerate(questions['follow_up_questions'], 1):
        print(f"{i}. {q['question']}")
        print(f"   Category: {q['category'].capitalize()}")
        print(f"   Importance: {q['importance']}")
        print()

def main():
    """Main function to run the website requirements analysis"""
    print("\n📋 WEBSITE REQUIREMENTS ANALYZER")
    print("="*80)
    print("This tool will help analyze website requirements from your description.")
    
    user_prompt = input("\n📝 Describe the website you want to build: ")
    
    print("\n⏳ Analyzing requirements...")
    try:
        # Step 1: Initial requirements analysis
        requirements = analyze_website_requirements(user_prompt)
        display_requirements(requirements)
        
        # Step 2: Check completeness
        assessment = check_requirements_completeness(requirements)
        display_assessment(assessment)
        
        # Step 3: Generate follow-up questions
        questions = generate_follow_up_questions(requirements, assessment)
        display_follow_up_questions(questions)
        
        # Step 4: Ask if user wants to refine requirements
        refine = input("\nWould you like to refine your requirements based on the follow-up questions? (y/n): ")
        
        if refine.lower() == 'y':
            # Collect answers to follow-up questions
            answers = {}
            for i, q in enumerate(questions['follow_up_questions'], 1):
                print(f"\nQuestion {i}: {q['question']}")
                answer = input("Your answer: ")
                answers[f"q{i}"] = answer
            
            # Refine requirements with answers
            refined_prompt = f"""
            Original website description: {user_prompt}
            
            Follow-up information:
            {json.dumps(answers)}
            
            Please provide updated and comprehensive website requirements based on all this information.
            """
            
            print("\n⏳ Refining requirements...")
            refined_requirements = analyze_website_requirements(refined_prompt)
            display_requirements(refined_requirements)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
if __name__ == "__main__":
    main()