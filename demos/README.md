# Campfires Framework Demos

This directory contains demonstration scripts that showcase the capabilities of the Campfires framework.

## Available Demos

### 1. Tax Application Team Demo (`tax_app_team_demo.py`)

A comprehensive software development team collaboration simulator that demonstrates advanced LLM integration with RAG (Retrieval-Augmented Generation) capabilities.

**Features:**
- **LLM-Powered Team Members**: Senior Backend Engineer, DevOps Engineer, Testing Engineer, and Frontend Developer
- **RAG Integration**: Team members have access to comprehensive documentation about the tax application system
- **Real LLM Responses**: Uses OpenRouter API with models like Claude-3.5-Sonnet for intelligent recommendations
- **Custom Prompt Engineering**: Implements `override_prompt` method for sophisticated LLM interactions
- **Professional Expertise**: Each team member provides role-specific insights and recommendations
- **HTML Report Generation**: Creates detailed meeting reports with actionable recommendations

**Technical Implementation:**
- **LLMCamperMixin**: Demonstrates proper integration of LLM capabilities into Camper classes
- **TeamMember Class**: Shows how to build intelligent agents with role-based expertise
- **OpenRouter Configuration**: Proper setup and usage of OpenRouter API for LLM calls
- **Error Handling**: Robust error handling for LLM API calls and network issues

**Team Roles and Expertise:**
- **Senior Backend Engineer**: API design, database optimization, security implementation
- **Senior DevOps Engineer**: Infrastructure, deployment, monitoring, scalability
- **Senior Testing Engineer**: Test strategies, automation, quality assurance
- **Senior Frontend Developer**: UI/UX design, user experience, accessibility

**To run:**
```bash
python demos/tax_app_team_demo.py
```

**Output:** Generates HTML reports with detailed team recommendations for software development decisions.

### 2. Hospital Zeitgeist Demo (`hospital_zeitgeist_demo.py`)

A sophisticated healthcare team collaboration simulator that demonstrates advanced multi-agent conversations with professional AI personas.

**Features:**
- **Professional Healthcare Characters**: Head Nurse, Admin Coordinator, Patient Advocate, IT Specialist, and Ward Manager
- **Zeitgeist Integration**: Real-time internet research for informed healthcare discussions
- **Action Planning**: Generates structured action plans with priorities, timelines, and responsible parties
- **HTML Reporting**: Creates detailed meeting reports with character responses and action items
- **Professional Personas**: Each character has realistic healthcare expertise and professional communication style
- **Dynamic Discussions**: Characters contribute based on their roles and expertise areas

**Healthcare Topics Covered:**
- Patient safety protocols
- Staff scheduling optimization
- Digital patient intake systems
- Emergency response procedures
- Patient feedback systems
- Medication safety protocols

**To run:**
```bash
python demos/hospital_zeitgeist_demo.py
```

**Output:** Generates HTML reports in the demos directory with complete meeting transcripts and action plans.

### 3. Simple Demo (`run_demo.py`)

A basic demonstration that shows core Campfires functionality without external dependencies.

**Features:**
- Text analysis (word count, sentiment, keyword detection)
- Text summarization
- Result logging to SQLite database
- Torch processing through multiple campers

**To run:**
```bash
python demos/run_demo.py
```

### 4. Reddit Crisis Tracker (`reddit_crisis_tracker.py`)

A comprehensive demo that simulates monitoring Reddit posts for mental health crisis situations.

**Features:**
- Mock Reddit API for generating crisis-related posts
- Crisis detection using keyword matching and LLM analysis
- Automated response generation for crisis posts
- Incident logging and tracking
- Integration with OpenRouter API for LLM capabilities

**Note:** This demo uses mock data and simulated API responses. To use with real APIs, you would need:
- Reddit API credentials (PRAW library)
- Valid OpenRouter API key
- Proper rate limiting and error handling

**To run:**
```bash
python demos/reddit_crisis_tracker.py
```

### 5. Zeitgeist Demo (`zeitgeist_demo.py`)

A demonstration of internet knowledge and opinion mining capabilities using the Zeitgeist integration.

**Features:**
- Real-time web search and information gathering
- Opinion mining from multiple sources
- Knowledge synthesis and analysis
- Integration with search engines for current information

**To run:**
```bash
python demos/zeitgeist_demo.py
```

## LLM Integration Patterns

The demos showcase several patterns for integrating Large Language Models into your Campfires applications:

### Pattern 1: LLMCamperMixin Integration

The most common pattern for adding LLM capabilities to your campers:

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig

class MyIntelligentCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        # Setup LLM configuration
        config = OpenRouterConfig(api_key="your-api-key")
        self.setup_llm(config)
    
    async def process(self, torch: Torch) -> Torch:
        # Use LLM for processing
        response = await self.llm_completion_with_mcp(f"Analyze: {torch.claim}")
        return Torch(claim=response, confidence=0.9)
```

### Pattern 2: Custom Prompt Engineering with override_prompt

For advanced LLM interactions with custom prompting strategies:

```python
class ExpertCamper(Camper, LLMCamperMixin):
    def override_prompt(self, torch: Torch) -> dict:
        """Custom prompt engineering for specialized responses"""
        enhanced_prompt = f"""
        You are an expert in {self.expertise}.
        Analyze: {torch.claim}
        Provide detailed insights and recommendations.
        """
        
        try:
            response = self.llm_completion_with_mcp(enhanced_prompt)
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {"expertise": self.expertise}
            }
        except Exception as e:
            return {
                "claim": f"Analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True}
            }
```

### Pattern 3: RAG-Enhanced Team Members

Combining document context with LLM reasoning for intelligent team collaboration:

```python
class TeamMember(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, rag_context: str):
        super().__init__(name)
        self.role = role
        self.rag_context = rag_context
    
    def override_prompt(self, torch: Torch) -> dict:
        """RAG-enhanced responses with role-specific expertise"""
        enhanced_prompt = f"""
        {self.rag_context}
        
        Role: {self.role}
        Question: {torch.claim}
        
        Provide detailed recommendations based on your role and context.
        """
        
        response = self.llm_completion_with_mcp(enhanced_prompt)
        return {
            "claim": response,
            "confidence": 0.9,
            "metadata": {"role": self.role, "rag_enhanced": True}
        }
```

## Character Examples

### Professional Healthcare Personas

The Hospital Zeitgeist Demo features professionally crafted AI characters:

**Sarah (Head Nurse)**
- Personality: Experienced clinical leader, patient-focused, detail-oriented
- Expertise: Clinical protocols, staff coordination, patient safety
- Communication Style: Professional, evidence-based, leadership-oriented

**Priya (Patient Advocate)**
- Personality: Dedicated advocate, patient-centered, quality-focused
- Expertise: Patient rights, healthcare accessibility, quality improvement
- Communication Style: Compassionate yet professional, equity-focused

**Dr. Elena (Ward Manager)**
- Personality: Strategic leader, evidence-based, operationally focused
- Expertise: Resource management, strategic planning, operational efficiency
- Communication Style: Data-driven, strategic, management-focused

**Liam (IT Specialist)**
- Personality: Quiet, tech-focused, solution-oriented
- Expertise: Healthcare technology, HIPAA compliance, system integration
- Communication Style: Technical, security-conscious, implementation-focused

### Action Planning Workflow

The system generates structured action plans with:

1. **Priority Levels**: High, Medium, Low based on urgency and impact
2. **Responsible Parties**: Specific roles assigned to each action item
3. **Timelines**: Realistic timeframes for implementation
4. **Dependencies**: Identification of prerequisite tasks
5. **Success Metrics**: Measurable outcomes for each action

**Example Action Plan Output:**
```
Priority: High
Action: Implement standardized medication verification protocol
Responsible: Head Nurse, Pharmacy Team
Timeline: 2 weeks
Dependencies: Staff training completion
Success Metric: 100% verification compliance rate
```

## Demo Architecture

Both demos follow the same Campfires architecture pattern:

1. **Torches**: Data containers that flow through the system
2. **Campers**: Processing units that transform torch data
3. **Campfire**: Orchestrator that manages campers and torch flow
4. **Box Driver**: Storage backend for assets and data
5. **State Manager**: Persistent state and logging
6. **MCP Protocol**: Message communication between components

## Output

When you run the demos, you'll see:
- Real-time processing logs
- Analysis results for each torch
- Summary statistics
- Database storage confirmation

## Extending the Demos

You can extend these demos by:
- Adding new camper types for different processing tasks
- Integrating with real APIs (Reddit, Twitter, etc.)
- Adding more sophisticated analysis algorithms
- Implementing different storage backends
- Creating custom MCP transport layers

## Requirements

The demos use only the core Campfires framework components. For the Reddit demo with real API integration, you would additionally need:
- `praw` for Reddit API
- `openai` or similar for LLM integration
- API keys and credentials