import requests
import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

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
    """Analyze the initial prompt and generate structured questions with options"""
    system_prompt = """
    You are an expert website requirements analyst. Your task is to:
    1. Analyze the user's website request
    2. Identify what critical information is missing
    3. Generate specific questions with multiple-choice or checkbox options to fill critical gaps
    
    Return your analysis as a structured JSON with:
    1. A summary of what is understood from the prompt
    2. A list of specific questions with pre-defined options to choose from
    
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
                "input_type": "radio|checkbox|dropdown",
                "options": [
                    {"id": "option_id_1", "text": "Option 1 text", "default": true/false},
                    {"id": "option_id_2", "text": "Option 2 text", "default": false}
                ],
                "critical_level": 1-5 (where 5 is most critical)
            }
        ]
    }
    
    Important: Include NO MORE THAN 5 questions, focusing only on the most critical information gaps.
    Order questions by critical_level (highest first).
    For each question:
    - Use radio buttons for mutually exclusive choices
    - Use checkboxes for "select all that apply" scenarios
    - Provide 3-6 relevant options per question
    - Mark one option as default where appropriate
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
                "key_elements": ["List of key elements on this page"],
                "detailed_functionality": "Comprehensive description of how this page functions and interacts with users"
            }
        ],
        "features": [
            {
                "name": "Feature name",
                "description": "Detailed description",
                "technical_details": "Specific implementation details, technologies, and approaches",
                "user_interaction": "How users will interact with this feature",
                "priority": "high|medium|low"
            }
        ],
        "design_requirements": {
            "style": "Overall style description",
            "color_scheme": "Description of colors",
            "typography": "Font preferences",
            "responsive_requirements": "How the site should behave on different devices",
            "accessibility_considerations": "Important accessibility features to implement"
        },
        "technical_specifications": {
            "platform": "Recommended platform/CMS",
            "integrations": ["Required external services"],
            "performance_requirements": "Speed/performance expectations",
            "security_requirements": "Security measures needed"
        },
        "third_party_solutions": [
            {
                "category": "Category (e.g., Email Marketing, Analytics, etc.)",
                "recommended_options": [
                    {
                        "name": "Solution name",
                        "description": "Brief description",
                        "integration_complexity": "low|medium|high",
                        "pricing_tier": "free|freemium|paid",
                        "best_for": "When this solution is most appropriate"
                    }
                ]
            }
        ],
        "content_requirements": [
            "List of content that needs to be created"
        ],
        "timeline": {
            "estimated_development_time": "Estimated time to build",
            "key_milestones": ["List of key milestones"],
            "potential_challenges": ["Anticipated challenges and how to address them"]
        },
        "maintenance_requirements": {
            "regular_updates": "Description of regular update needs",
            "ongoing_content": "Content management strategy",
            "technical_maintenance": "Technical maintenance needs"
        }
    }
    
    IMPORTANT GUIDELINES:
    1. Be extremely detailed and specific in the features section, covering all aspects of functionality
    2. For third-party solutions, recommend 2-3 specific tools for each relevant category (e.g., email marketing, analytics, payment processing, CRM, etc.)
    3. Consider the most appropriate solutions based on the website's purpose, audience, and technical requirements
    4. Include both popular industry standards and potentially novel or specialized solutions that might be particularly suitable
    5. For each page, provide comprehensive details about functionality and user interactions
    """
    
    combined_prompt = f"""
    Initial website request:
    {initial_prompt}
    
    Additional information from follow-up questions:
    {json.dumps(answers, indent=2)}
    
    Based on all this information, create comprehensive website requirements.
    """
    
    return call_azure_openai(system_prompt, combined_prompt, temperature=0.4)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    initial_prompt = data.get('prompt', '')
    
    try:
        analysis = analyze_initial_prompt(initial_prompt)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    initial_prompt = data.get('prompt', '')
    answers = data.get('answers', {})
    
    try:
        requirements = create_comprehensive_requirements(initial_prompt, answers)
        return jsonify(requirements)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Website Requirements Generator</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .loader {
            border-top-color: #3498db;
            -webkit-animation: spinner 1.5s linear infinite;
            animation: spinner 1.5s linear infinite;
        }
        @-webkit-keyframes spinner {
            0% { -webkit-transform: rotate(0deg); }
            100% { -webkit-transform: rotate(360deg); }
        }
        @keyframes spinner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .fade-in {
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        .option-card {
            transition: all 0.2s ease;
        }
        .option-card:hover {
            transform: translateY(-2px);
        }
        .option-card.selected {
            border-color: #4299e1;
            background-color: #ebf8ff;
        }
        .option-card input {
            position: absolute;
            opacity: 0;
            cursor: pointer;
            height: 0;
            width: 0;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="text-center mb-10">
            <h1 class="text-3xl font-bold text-gray-800">Website Requirements Generator</h1>
            <p class="text-gray-600 mt-2">Generate comprehensive website plans with minimal input</p>
        </header>

        <div id="step1" class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-xl font-semibold mb-4">Step 1: Describe Your Website</h2>
            <div class="mb-4">
                <label for="prompt" class="block text-gray-700 mb-2">What kind of website do you want to build?</label>
                <textarea id="prompt" rows="4" class="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400" placeholder="Describe your website idea as thoroughly as possible..."></textarea>
            </div>
            <button id="analyzeBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition">
                Analyze Requirements
            </button>
            <div id="analyzeLoader" class="hidden flex justify-center items-center mt-4">
                <div class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-10 w-10"></div>
                <span class="ml-2">Analyzing your request...</span>
            </div>
        </div>

        <div id="step2" class="hidden bg-white rounded-lg shadow-md p-6 mb-6 fade-in">
            <h2 class="text-xl font-semibold mb-4">Step 2: Additional Information Needed</h2>
            <div id="understood" class="mb-6 p-4 bg-green-50 rounded-md">
                <h3 class="font-medium text-green-800 mb-2">Here's what I understood from your description:</h3>
                <div id="understoodContent" class="text-green-700"></div>
            </div>
            <div id="questions" class="mb-6">
                <h3 class="font-medium mb-4">To complete your website requirements, please answer these questions:</h3>
                <div id="questionsContent" class="space-y-8"></div>
            </div>
            <div class="mt-6">
                <button id="generateBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md transition">
                    Generate Website Plan
                </button>
                <div id="generateLoader" class="hidden flex justify-center items-center mt-4">
                    <div class="loader ease-linear rounded-full border-4 border-t-4 border-gray-200 h-10 w-10"></div>
                    <span class="ml-2">Generating your website plan...</span>
                </div>
            </div>
        </div>

        <div id="step3" class="hidden bg-white rounded-lg shadow-md p-6 mb-6 fade-in">
            <h2 class="text-xl font-semibold mb-4">Step 3: Your Website Plan</h2>
            <div id="requirements" class="space-y-6">
                <!-- Requirements will be populated here -->
            </div>
            <div class="mt-6 flex justify-between">
                <button id="backBtn" class="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-4 rounded-md transition">
                    Back to Questions
                </button>
                <button id="downloadBtn" class="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition">
                    Download Plan as JSON
                </button>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let currentPrompt = '';
        let currentAnalysis = null;
        let currentRequirements = null;

        // DOM elements
        const analyzeBtn = document.getElementById('analyzeBtn');
        const generateBtn = document.getElementById('generateBtn');
        const backBtn = document.getElementById('backBtn');
        const downloadBtn = document.getElementById('downloadBtn');
        const step1 = document.getElementById('step1');
        const step2 = document.getElementById('step2');
        const step3 = document.getElementById('step3');
        const analyzeLoader = document.getElementById('analyzeLoader');
        const generateLoader = document.getElementById('generateLoader');
        const promptInput = document.getElementById('prompt');
        const understoodContent = document.getElementById('understoodContent');
        const questionsContent = document.getElementById('questionsContent');
        const requirementsDiv = document.getElementById('requirements');

        // Event listeners
        analyzeBtn.addEventListener('click', analyzePrompt);
        generateBtn.addEventListener('click', generateRequirements);
        backBtn.addEventListener('click', goBackToQuestions);
        downloadBtn.addEventListener('click', downloadRequirements);

        // Analyze prompt
        async function analyzePrompt() {
            currentPrompt = promptInput.value.trim();
            
            if (!currentPrompt) {
                alert('Please describe your website first.');
                return;
            }
            
            // Show loader
            analyzeBtn.classList.add('hidden');
            analyzeLoader.classList.remove('hidden');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ prompt: currentPrompt }),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                currentAnalysis = data;
                displayAnalysis(data);
                
                // Show step 2
                step1.classList.add('hidden');
                step2.classList.remove('hidden');
                
            } catch (error) {
                console.error('Error analyzing prompt:', error);
                alert('Error analyzing your request. Please try again.');
                analyzeBtn.classList.remove('hidden');
                analyzeLoader.classList.add('hidden');
            }
        }

        // Display analysis results
        function displayAnalysis(analysis) {
            // Display what we understood
            let understoodHTML = '';
            for (const [key, value] of Object.entries(analysis.understood)) {
                if (value && value !== 'null' && value.length > 0) {
                    const formattedKey = key.replace('_', ' ').charAt(0).toUpperCase() + key.replace('_', ' ').slice(1);
                    if (Array.isArray(value)) {
                        understoodHTML += `<p><strong>${formattedKey}:</strong> ${value.join(', ')}</p>`;
                    } else {
                        understoodHTML += `<p><strong>${formattedKey}:</strong> ${value}</p>`;
                    }
                }
            }
            understoodContent.innerHTML = understoodHTML || '<p>I need more information about your website.</p>';
            
            // Display questions with interactive options
            let questionsHTML = '';
            if (analysis.questions && analysis.questions.length > 0) {
                analysis.questions.forEach((q, index) => {
                    let optionsHTML = '';
                    
                    if (q.input_type === 'radio') {
                        optionsHTML = `
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
                                ${q.options.map(option => `
                                    <label class="option-card relative flex flex-col p-4 border rounded-md cursor-pointer">
                                        <input type="radio" name="question-${q.id}" data-id="${q.id}" data-option-id="${option.id}" ${option.default ? 'checked' : ''} />
                                        <span class="font-medium">${option.text}</span>
                                    </label>
                                `).join('')}
                            </div>
                        `;
                    } else if (q.input_type === 'checkbox') {
                        optionsHTML = `
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-3">
                                ${q.options.map(option => `
                                    <label class="option-card relative flex flex-col p-4 border rounded-md cursor-pointer">
                                        <input type="checkbox" name="question-${q.id}" data-id="${q.id}" data-option-id="${option.id}" ${option.default ? 'checked' : ''} />
                                        <span class="font-medium">${option.text}</span>
                                    </label>
                                `).join('')}
                            </div>
                        `;
                    } else if (q.input_type === 'dropdown') {
                        optionsHTML = `
                            <select class="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 mt-3" data-id="${q.id}">
                                ${q.options.map(option => `
                                    <option value="${option.id}" ${option.default ? 'selected' : ''}>${option.text}</option>
                                `).join('')}
                            </select>
                        `;
                    }
                    
                    questionsHTML += `
                        <div class="p-5 border rounded-md bg-white">
                            <h4 class="text-lg font-medium text-gray-800">${index + 1}. ${q.question}</h4>
                            <p class="text-sm text-gray-500 mb-3">Category: ${q.category}</p>
                            ${optionsHTML}
                        </div>
                    `;
                });
            } else {
                questionsHTML = '<p class="text-green-700 font-medium">Your description was comprehensive! No additional questions needed.</p>';
            }
            questionsContent.innerHTML = questionsHTML;
            
            // Add event listeners to option cards for UI feedback
            document.querySelectorAll('.option-card').forEach(card => {
                const input = card.querySelector('input');
                
                // Initialize selected state
                if (input.checked) {
                    card.classList.add('selected');
                }
                
                input.addEventListener('change', () => {
                    if (input.type === 'radio') {
                        // Remove selected class from all options in the same group
                        document.querySelectorAll(`input[name="${input.name}"]`).forEach(radio => {
                            radio.closest('.option-card').classList.remove('selected');
                        });
                    }
                    
                    // Toggle selected class based on checked state
                    if (input.checked) {
                        card.classList.add('selected');
                    } else {
                        card.classList.remove('selected');
                    }
                });
            });
        }

        // Generate final requirements
        async function generateRequirements() {
            // Collect answers from radio, checkbox, and dropdown inputs
            const answers = {};
            
            // Process radio inputs
            document.querySelectorAll('input[type="radio"]:checked').forEach(radio => {
                const questionId = radio.dataset.id;
                const optionId = radio.dataset.optionId;
                const question = currentAnalysis.questions.find(q => q.id === questionId);
                const option = question.options.find(o => o.id === optionId);
                
                answers[questionId] = {
                    question: question.question,
                    answer: option.text,
                    category: question.category
                };
            });
            
            // Process checkbox inputs (group by question id)
            const checkboxGroups = {};
            document.querySelectorAll('input[type="checkbox"]:checked').forEach(checkbox => {
                const questionId = checkbox.dataset.id;
                const optionId = checkbox.dataset.optionId;
                const question = currentAnalysis.questions.find(q => q.id === questionId);
                const option = question.options.find(o => o.id === optionId);
                
                if (!checkboxGroups[questionId]) {
                    checkboxGroups[questionId] = {
                        question: question.question,
                        answer: [],
                        category: question.category
                    };
                }
                
                checkboxGroups[questionId].answer.push(option.text);
            });
            
            // Convert checkbox groups to string answers
            Object.entries(checkboxGroups).forEach(([questionId, data]) => {
                answers[questionId] = {
                    question: data.question,
                    answer: data.answer.join(', '),
                    category: data.category
                };
            });
            
            // Process dropdown selects
            document.querySelectorAll('select').forEach(select => {
                const questionId = select.dataset.id;
                const optionId = select.value;
                const question = currentAnalysis.questions.find(q => q.id === questionId);
                const option = question.options.find(o => o.id === optionId);
                
                answers[questionId] = {
                    question: question.question,
                    answer: option.text,
                    category: question.category
                };
            });
            
            // Show loader
            generateBtn.classList.add('hidden');
            generateLoader.classList.remove('hidden');
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: currentPrompt,
                        answers: answers
                    }),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                currentRequirements = data;
                displayRequirements(data);
                
                // Show step 3
                step2.classList.add('hidden');
                step3.classList.remove('hidden');
                
            } catch (error) {
                console.error('Error generating requirements:', error);
                alert('Error generating your website plan. Please try again.');
                generateBtn.classList.remove('hidden');
                generateLoader.classList.add('hidden');
            }
        }

        // Display final requirements
        function displayRequirements(requirements) {
            let html = '';
            
            // Website Summary
            const summary = requirements.website_summary;
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Website Summary</h3>
                    <div class="bg-blue-50 p-4 rounded-md">
                        <p class="mb-2"><strong>Name:</strong> ${summary.name}</p>
                        <p class="mb-2"><strong>Purpose:</strong> ${summary.purpose}</p>
                        <p><strong>Target Audience:</strong> ${summary.target_audience}</p>
                    </div>
                </div>
            `;
            
            // Pages
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Pages</h3>
                    <div class="grid grid-cols-1 gap-4">
            `;
            
            requirements.pages.forEach((page, index) => {
                html += `
                    <div class="p-4 border rounded-md">
                        <h4 class="font-medium text-lg">${index + 1}. ${page.name}</h4>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                            <div>
                                <p class="text-sm mb-2"><strong>Purpose:</strong> ${page.purpose}</p>
                                <div class="text-sm">
                                    <p class="font-medium mb-1">Key Elements:</p>
                                    <ul class="list-disc pl-5">
                                        ${page.key_elements.map(el => `<li>${el}</li>`).join('')}
                                    </ul>
                                </div>
                            </div>
                            <div>
                                <p class="text-sm font-medium mb-1">Detailed Functionality:</p>
                                <p class="text-sm">${page.detailed_functionality || 'Not specified'}</p>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
            
            // Features
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Features</h3>
                    <div class="space-y-4">
            `;
            
            requirements.features.forEach((feature, index) => {
                const priorityColors = {
                    'high': 'bg-red-100 text-red-800',
                    'medium': 'bg-yellow-100 text-yellow-800',
                    'low': 'bg-green-100 text-green-800'
                };
                
                const priorityColor = priorityColors[feature.priority] || 'bg-gray-100 text-gray-800';
                
                html += `
                    <div class="p-4 border rounded-md">
                        <div class="flex justify-between items-center mb-2">
                            <h4 class="font-medium text-lg">${index + 1}. ${feature.name}</h4>
                            <span class="px-2 py-1 text-xs rounded-full ${priorityColor}">
                                ${feature.priority} priority
                            </span>
                        </div>
                        <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <p class="text-sm font-medium mb-1">Description:</p>
                                <p class="text-sm">${feature.description}</p>
                            </div>
                            <div>
                                <p class="text-sm font-medium mb-1">Technical Details:</p>
                                <p class="text-sm">${feature.technical_details || 'Not specified'}</p>
                            </div>
                        </div>
                        <div class="mt-3">
                            <p class="text-sm font-medium mb-1">User Interaction:</p>
                            <p class="text-sm">${feature.user_interaction || 'Not specified'}</p>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
            
            // Design Requirements
            const design = requirements.design_requirements;
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Design Requirements</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="p-4 border rounded-md">
                            <h4 class="font-medium">Style</h4>
                            <p class="text-sm">${design.style}</p>
                        </div>
                        <div class="p-4 border rounded-md">
                            <h4 class="font-medium">Color Scheme</h4>
                            <p class="text-sm">${design.color_scheme}</p>
                        </div>
                        <div class="p-4 border rounded-md">
                            <h4 class="font-medium">Typography</h4>
                            <p class="text-sm">${design.typography}</p>
                        </div>
                        <div class="p-4 border rounded-md">
                            <h4 class="font-medium">Responsive Design</h4>
                            <p class="text-sm">${design.responsive_requirements}</p>
                        </div>
                        <div class="p-4 border rounded-md col-span-1 md:col-span-2">
                            <h4 class="font-medium">Accessibility Considerations</h4>
                            <p class="text-sm">${design.accessibility_considerations || 'Not specified'}</p>
                        </div>
                    </div>
                </div>
            `;
            
            // Technical Specifications
            const tech = requirements.technical_specifications;
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Technical Specifications</h3>
                    <div class="p-4 border rounded-md">
                        <p class="mb-2"><strong>Recommended Platform:</strong> ${tech.platform}</p>
                        <p class="mb-2"><strong>Integrations:</strong> ${tech.integrations.join(', ') || 'None'}</p>
                        <p class="mb-2"><strong>Performance Requirements:</strong> ${tech.performance_requirements}</p>
                        <p><strong>Security Requirements:</strong> ${tech.security_requirements || 'Not specified'}</p>
                    </div>
                </div>
            `;
            
            // Third-Party Solutions
            if (requirements.third_party_solutions && requirements.third_party_solutions.length > 0) {
                html += `
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-2">Recommended Third-Party Solutions</h3>
                        <div class="space-y-4">
                `;
                
                requirements.third_party_solutions.forEach(category => {
                    html += `
                        <div class="p-4 border rounded-md">
                            <h4 class="font-medium text-lg mb-3">${category.category}</h4>
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    `;
                    
                    category.recommended_options.forEach(option => {
                        const complexityColors = {
                            'low': 'bg-green-100 text-green-800',
                            'medium': 'bg-yellow-100 text-yellow-800',
                            'high': 'bg-red-100 text-red-800'
                        };
                        
                        const pricingColors = {
                            'free': 'bg-green-100 text-green-800',
                            'freemium': 'bg-blue-100 text-blue-800',
                            'paid': 'bg-purple-100 text-purple-800'
                        };
                        
                        const complexityColor = complexityColors[option.integration_complexity] || 'bg-gray-100 text-gray-800';
                        const pricingColor = pricingColors[option.pricing_tier] || 'bg-gray-100 text-gray-800';
                        
                        html += `
                            <div class="p-3 border rounded-md bg-white shadow-sm">
                                <div class="flex justify-between items-start">
                                    <h5 class="font-medium">${option.name}</h5>
                                    <div class="flex space-x-1">
                                        <span class="px-2 py-1 text-xs rounded-full ${complexityColor}">
                                            ${option.integration_complexity}
                                        </span>
                                        <span class="px-2 py-1 text-xs rounded-full ${pricingColor}">
                                            ${option.pricing_tier}
                                        </span>
                                    </div>
                                </div>
                                <p class="text-sm mt-2">${option.description}</p>
                                <p class="text-sm mt-2"><strong>Best for:</strong> ${option.best_for}</p>
                            </div>
                        `;
                    });
                    
                    html += `
                            </div>
                        </div>
                    `;
                });
                
                html += `
                        </div>
                    </div>
                `;
            }
            
            // Content Requirements
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Content Requirements</h3>
                    <div class="p-4 border rounded-md">
                        <ul class="list-disc pl-5">
                            ${requirements.content_requirements.map(content => `<li>${content}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
            
            // Timeline
            const timeline = requirements.timeline;
            html += `
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Timeline</h3>
                    <div class="p-4 border rounded-md">
                        <p class="mb-2"><strong>Estimated Development Time:</strong> ${timeline.estimated_development_time}</p>
                        <div class="mb-4">
                            <p class="font-medium mb-1">Key Milestones:</p>
                            <ol class="list-decimal pl-5">
                                ${timeline.key_milestones.map(milestone => `<li>${milestone}</li>`).join('')}
                            </ol>
                        </div>
                        ${timeline.potential_challenges && timeline.potential_challenges.length > 0 ? `
                            <div>
                                <p class="font-medium mb-1">Potential Challenges:</p>
                                <ul class="list-disc pl-5">
                                    ${timeline.potential_challenges.map(challenge => `<li>${challenge}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            // Maintenance Requirements
            if (requirements.maintenance_requirements) {
                const maintenance = requirements.maintenance_requirements;
                html += `
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-2">Maintenance Requirements</h3>
                        <div class="p-4 border rounded-md">
                            <p class="mb-2"><strong>Regular Updates:</strong> ${maintenance.regular_updates || 'Not specified'}</p>
                            <p class="mb-2"><strong>Ongoing Content:</strong> ${maintenance.ongoing_content || 'Not specified'}</p>
                            <p><strong>Technical Maintenance:</strong> ${maintenance.technical_maintenance || 'Not specified'}</p>
                        </div>
                    </div>
                `;
            }
            
            requirementsDiv.innerHTML = html;
        }

        // Go back to questions
        function goBackToQuestions() {
            step3.classList.add('hidden');
            step2.classList.remove('hidden');
            generateBtn.classList.remove('hidden');
            generateLoader.classList.add('hidden');
        }

        // Download requirements as JSON
        function downloadRequirements() {
            if (!currentRequirements) return;
            
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentRequirements, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "website_requirements.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
        }
    </script>
</body>
</html>
        ''')
    
    # Run the app
    app.run(debug=True)