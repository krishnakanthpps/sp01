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

def call_azure_openai(system_prompt, user_prompt, temperature=0.3):
    """Generic function to call Azure OpenAI API"""
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    data = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return json.loads(response.json()["choices"][0]["message"]["content"])
    else:
        raise Exception(f"Error calling Azure OpenAI API: {response.text}")

def analyze_initial_prompt(prompt):
    """Analyze the initial prompt and identify what information is missing"""
    system_prompt = """
    You are an expert website requirements analyst. Your task is to:
    1. Analyze the user's website request
    2. Identify what critical information is missing
    3. Generate the minimum number of questions needed to complete the requirements
    
    Return your analysis as a structured JSON with:
    1. A summary of what is understood from the prompt
    2. A list of specific questions to fill critical gaps
    
    Your response must be in this format:
    {
        "understood": {
            "purpose": "What you understand about the website's purpose (or null if unclear)",
            "audience": "What you understand about the target audience (or null if unclear)",
            "features": ["List of features you can identify from the prompt"],
            "design_preferences": "What you understand about design preferences (or null if unclear)"
        },
        "questions": [
            {
                "id": "unique_question_id",
                "question": "Clear, specific question text",
                "category": "purpose|audience|features|design|technical",
                "critical_level": 1-5 (where 5 is most critical)
            }
        ]
    }
    
    Important: Include NO MORE THAN 5 questions, focusing only on the most critical information gaps.
    Order questions by critical_level (highest first).
    """
    
    return call_azure_openai(system_prompt, prompt)

def create_comprehensive_requirements(initial_prompt, answers):
    """Generate comprehensive website requirements based on initial prompt and answers"""
    system_prompt = """
    You are a website project planner who creates detailed website specifications.
    
    Based on the initial website request and the follow-up answers provided, create a comprehensive website requirements document.
    
    Return your requirements as structured JSON with these sections:
    {
        "website_summary": {
            "name": "Name of the website",
            "purpose": "Clear statement of the website's purpose",
            "target_audience": "Description of who will use the website"
        },
        "pages": [
            {
                "name": "Name of the page",
                "purpose": "Purpose of this page",
                "key_elements": ["List of key elements on this page"]
            }
        ],
        "features": [
            {
                "name": "Feature name",
                "description": "Detailed description",
                "priority": "high|medium|low"
            }
        ],
        "design_requirements": {
            "style": "Overall style description",
            "color_scheme": "Description of colors",
            "typography": "Font preferences",
            "responsive_requirements": "How the site should behave on different devices"
        },
        "technical_specifications": {
            "platform": "Recommended platform/CMS",
            "integrations": ["Required external services"],
            "performance_requirements": "Speed/performance expectations"
        },
        "content_requirements": [
            "List of content that needs to be created"
        ],
        "timeline": {
            "estimated_development_time": "Estimated time to build",
            "key_milestones": ["List of key milestones"]
        }
    }
    """
    
    combined_prompt = f"""
    Initial website request:
    {initial_prompt}
    
    Additional information from follow-up questions:
    {json.dumps(answers, indent=2)}
    
    Based on all this information, create comprehensive website requirements.
    """
    
    return call_azure_openai(system_prompt, combined_prompt, temperature=0.4)

def display_formatted_requirements(requirements):
    """Display the requirements in a nicely formatted way"""
    print("\n" + "="*80)
    print(f"🌐 WEBSITE REQUIREMENTS: {requirements['website_summary']['name']}")
    print("="*80)
    
    # Website Summary
    summary = requirements['website_summary']
    print(f"\n📋 WEBSITE SUMMARY")
    print(f"  Purpose: {summary['purpose']}")
    print(f"  Target Audience: {summary['target_audience']}")
    
    # Pages
    print(f"\n📄 PAGES")
    for i, page in enumerate(requirements['pages'], 1):
        print(f"  {i}. {page['name']}")
        print(f"     Purpose: {page['purpose']}")
        print(f"     Key Elements:")
        for element in page['key_elements']:
            print(f"       • {element}")
    
    # Features
    print(f"\n⚙️ FEATURES")
    for i, feature in enumerate(requirements['features'], 1):
        print(f"  {i}. {feature['name']} (Priority: {feature['priority']})")
        print(f"     {feature['description']}")
    
    # Design Requirements
    print(f"\n🎨 DESIGN REQUIREMENTS")
    design = requirements['design_requirements']
    print(f"  Style: {design['style']}")
    print(f"  Color Scheme: {design['color_scheme']}")
    print(f"  Typography: {design['typography']}")
    print(f"  Responsive Requirements: {design['responsive_requirements']}")
    
    # Technical Specifications
    print(f"\n🔧 TECHNICAL SPECIFICATIONS")
    tech = requirements['technical_specifications']
    print(f"  Platform: {tech['platform']}")
    print(f"  Integrations: {', '.join(tech['integrations'])}")
    print(f"  Performance Requirements: {tech['performance_requirements']}")
    
    # Content Requirements
    print(f"\n📝 CONTENT REQUIREMENTS")
    for i, content in enumerate(requirements['content_requirements'], 1):
        print(f"  {i}. {content}")
    
    # Timeline
    print(f"\n⏱️ TIMELINE")
    timeline = requirements['timeline']
    print(f"  Estimated Development Time: {timeline['estimated_development_time']}")
    print(f"  Key Milestones:")
    for milestone in timeline['key_milestones']:
        print(f"    • {milestone}")

def main():
    print("\n🔍 WEBSITE REQUIREMENTS ANALYZER")
    print("="*80)
    
    # Get initial prompt
    initial_prompt = input("\n📝 Describe the website you want to build: ")
    
    # Analyze prompt and generate focused questions
    print("\n⏳ Analyzing your request...")
    analysis = analyze_initial_prompt(initial_prompt)
    
    # Display what was understood
    print("\n✅ Here's what I understood from your description:")
    for key, value in analysis['understood'].items():
        if value and value != "null":
            formatted_key = key.replace('_', ' ').capitalize()
            if isinstance(value, list):
                print(f"  {formatted_key}: {', '.join(value)}")
            else:
                print(f"  {formatted_key}: {value}")
    
    # If there are questions, ask them
    if analysis['questions']:
        print("\n❓ I need a bit more information to complete the requirements:")
        
        # Collect answers
        answers = {}
        for i, q in enumerate(analysis['questions'], 1):
            print(f"\n{i}. {q['question']}")
            answer = input("   Your answer: ")
            answers[q['id']] = {
                "question": q['question'],
                "answer": answer,
                "category": q['category']
            }
        
        # Generate comprehensive requirements
        print("\n⏳ Generating comprehensive website requirements...")
        requirements = create_comprehensive_requirements(initial_prompt, answers)
        
        # Display formatted requirements
        display_formatted_requirements(requirements)
        
        # Ask if they want to save the requirements
        save = input("\nWould you like to save these requirements to a file? (y/n): ")
        if save.lower() == 'y':
            filename = input("Enter filename (default: website_requirements.json): ") or "website_requirements.json"
            with open(filename, 'w') as f:
                json.dump(requirements, f, indent=2)
            print(f"\n✅ Requirements saved to {filename}")
    
    else:
        print("\n✅ Your initial description was comprehensive! No additional questions needed.")
        # Generate requirements directly from the initial prompt
        requirements = create_comprehensive_requirements(initial_prompt, {})
        display_formatted_requirements(requirements)

if __name__ == "__main__":
    main()