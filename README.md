# Campfires Framework

A Python framework for orchestrating multimodal Large Language Models (LLMs) and tools to achieve emergent, task-driven behavior.

![Campfires Logo](images/logo.jpg)

## The Valley of Campfires

Imagine a peaceful valley at twilight, dotted with glowing campfires. Around each campfire, a group of **Campers** (AI agents) sit together, sharing stories, analyzing information, and working on tasks. Each campfire represents a **Campfire** - a collaborative workspace where agents can communicate and coordinate their efforts.

### The Campfire Community

At your campfire, **Campers** pass around **Torches** - glowing vessels that carry information, data, and insights from one agent to another. Each torch illuminates the conversation, bringing new perspectives and knowledge to the circle. As campers examine and discuss what each torch reveals, they add their own insights, transforming the information before passing it along.

### The Party Box Exchange

Between the campfires sits a magical **Party Box** - a shared storage space where campfires can exchange gifts, artifacts, and resources. When your campers discover something valuable (documents, images, audio files, or data), they can place it in the Party Box for other campfires to discover and use. It's like a community treasure chest that connects all the campfires in the valley.

![The Valley of Campfires](images/campfires.jpg)
*A peaceful valley at twilight, where AI agents gather around glowing campfires to collaborate, share knowledge through torches, and exchange resources via the central Party Box. Each campfire represents a collaborative workspace, while the glowing Party Box in the center connects all communities across the valley.*

### The Torch Bearer Network

When something important happens at your campfire - a breakthrough discovery, a completed task, or an urgent message - a **Torch Bearer** can carry the news to other campfires throughout the valley. These torch bearers use the **MCP Protocol** (Model Context Protocol) to deliver information packets, ensuring that all campfires stay connected and informed about events, notifications, and shared resources.

### Your Valley, Your Rules

Each campfire operates independently, with its own group of specialized campers, but they're all part of the same vibrant valley community. Whether you're running a single intimate campfire or orchestrating multiple campfires across the valley, the framework provides the tools to create emergent, collaborative AI behaviors that feel as natural as friends gathering around a fire.

Welcome to the valley. Pull up a log, grab a torch, and let's build something amazing together.

## Features

- **Modular Architecture**: Build complex AI workflows using composable "Campers" (AI agents)
- **LLM Integration**: Built-in support for OpenRouter and other LLM providers
- **Enhanced Orchestration**: Advanced task orchestration with detailed execution stages, problem understanding, approach selection, and quality considerations
- **Interactive HTML Reports**: Rich HTML reports with expandable sections showing execution stages, RAG information, customization details, and impact analysis
- **Zeitgeist**: Internet knowledge and opinion mining for informed campers
- **Action Planning**: Generate structured action plans with priorities and timelines
- **Professional Character System**: Define unique personalities and perspectives with professional traits
- **RAG Integration**: Retrieval-Augmented Generation with document context and state management
- **MCP Protocol**: Model Context Protocol for inter-agent communication
- **Storage Management**: Flexible "Party Box" system for asset storage
- **State Management**: Persistent state tracking with SQLite backend
- **Template System**: Dynamic prompt templating with Jinja2

## Installation

### From PyPI (Recommended)

```bash
pip install campfires
```

### From Source

```bash
git clone https://github.com/campfires/campfires.git
cd campfires
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from campfires import Campfire, Camper, Torch, OpenRouterConfig, LLMCamperMixin

class MyCamper(Camper, LLMCamperMixin):
    async def process(self, torch: Torch) -> Torch:
        # Process the input torch and return a new torch
        response = await self.llm_completion(f"Analyze: {torch.claim}")
        return Torch(
            claim=response,
            confidence=0.8,
            metadata={"processed_by": "MyCamper"}
        )

async def main():
    # Setup LLM configuration
    config = OpenRouterConfig(
        api_key="your-openrouter-api-key",
        default_model="anthropic/claude-3-sonnet"
    )
    
    # Create camper and setup LLM
    camper = MyCamper("my-camper")
    camper.setup_llm(config)
    
    # Create campfire and add camper
    campfire = Campfire("my-campfire")
    campfire.add_camper(camper)
    
    # Start the campfire
    await campfire.start()
    
    # Send a torch for processing
    input_torch = Torch(claim="Hello, world!")
    await campfire.send_torch(input_torch)
    
    # Stop the campfire
    await campfire.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Crisis Detection Example

```python
import asyncio
from campfires import (
    Campfire, Camper, Torch, 
    OpenRouterConfig, LLMCamperMixin,
    MCPProtocol, AsyncQueueTransport
)

class CrisisDetectionCamper(Camper, LLMCamperMixin):
    async def process(self, torch: Torch) -> Torch:
        # Analyze text for crisis indicators
        prompt = f"""
        Analyze this text for crisis indicators:
        "{torch.claim}"
        
        Return JSON with crisis_probability (0-1) and key_indicators.
        """
        
        response = await self.llm_completion_with_mcp(
            prompt, 
            channel="crisis_detection"
        )
        
        return Torch(
            claim=f"Crisis analysis: {response}",
            confidence=0.9,
            metadata={"analysis_type": "crisis_detection"}
        )

async def main():
    # Setup MCP protocol for inter-camper communication
    transport = AsyncQueueTransport()
    mcp_protocol = MCPProtocol(transport)
    await mcp_protocol.start()
    
    # Setup LLM configuration
    config = OpenRouterConfig(
        api_key="your-openrouter-api-key",
        default_model="anthropic/claude-3-sonnet"
    )
    
    # Create and configure camper
    camper = CrisisDetectionCamper("crisis-detector")
    camper.setup_llm(config, mcp_protocol)
    
    # Create campfire with MCP support
    campfire = Campfire("crisis-campfire", mcp_protocol=mcp_protocol)
    campfire.add_camper(camper)
    
    await campfire.start()
    
    # Process some text
    torch = Torch(claim="I'm feeling really overwhelmed and don't know what to do")
    await campfire.send_torch(torch)
    
    await campfire.stop()
    await mcp_protocol.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### LLM-Enabled Campers with Custom Prompts

The framework supports advanced LLM integration through the `override_prompt` method, allowing campers to customize their LLM interactions:

```python
import asyncio
from campfires import Camper, Torch, OpenRouterConfig, LLMCamperMixin

class ExpertAnalyzer(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str):
        super().__init__(name)
        self.expertise = expertise
        
    def override_prompt(self, torch: Torch) -> dict:
        """Custom prompt generation with LLM call"""
        try:
            # Create enhanced prompt based on expertise
            enhanced_prompt = f"""
            You are an expert {self.expertise}. Analyze the following information 
            and provide professional insights:
            
            Input: {torch.claim}
            
            Please provide:
            1. Key insights from your {self.expertise} perspective
            2. Potential concerns or opportunities
            3. Recommended next steps
            """
            
            # Make LLM call directly in override_prompt
            response = self.llm_completion_with_mcp(enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "expertise": self.expertise,
                    "analysis_type": "expert_review"
                }
            }
        except Exception as e:
            return {
                "claim": f"Analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True}
            }

async def main():
    # Setup LLM configuration
    config = OpenRouterConfig(api_key="your-openrouter-api-key")
    
    # Create expert campers
    security_expert = ExpertAnalyzer("security-expert", "cybersecurity")
    security_expert.setup_llm(config)
    
    finance_expert = ExpertAnalyzer("finance-expert", "financial analysis")
    finance_expert.setup_llm(config)
    
    # Create campfire and add experts
    campfire = Campfire("expert-analysis")
    campfire.add_camper(security_expert)
    campfire.add_camper(finance_expert)
    
    await campfire.start()
    
    # Analyze a business proposal
    torch = Torch(claim="We're considering implementing a new payment system")
    await campfire.send_torch(torch)
    
    await campfire.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Team Collaboration with RAG Integration

Build intelligent team members that can access and reason over document collections:

```python
import asyncio
from campfires import Camper, Torch, OpenRouterConfig, LLMCamperMixin

class TeamMember(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, rag_system_prompt: str):
        super().__init__(name)
        self.role = role
        self.rag_system_prompt = rag_system_prompt
        
    def override_prompt(self, torch: Torch) -> dict:
        """Generate responses using RAG-enhanced prompts"""
        try:
            # Combine RAG context with user question
            enhanced_prompt = f"""
            {self.rag_system_prompt}
            
            Role: {self.role}
            Question: {torch.claim}
            
            Please provide a detailed response based on your role and the 
            available context. Include specific recommendations and actionable insights.
            """
            
            # Make LLM call with enhanced context
            response = self.llm_completion_with_mcp(enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "role": self.role,
                    "rag_enhanced": True,
                    "response_type": "team_recommendation"
                }
            }
        except Exception as e:
            return {
                "claim": f"Unable to provide recommendation: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "role": self.role}
            }

async def main():
    # Setup LLM configuration
    config = OpenRouterConfig(api_key="your-openrouter-api-key")
    
    # RAG system prompt with document context
    rag_context = """
    You have access to comprehensive documentation about our tax application system.
    The system handles tax calculations, user management, and compliance reporting.
    Key components include: authentication service, calculation engine, reporting module.
    """
    
    # Create team members with different roles
    backend_engineer = TeamMember(
        "backend-engineer", 
        "Senior Backend Engineer",
        rag_context
    )
    backend_engineer.setup_llm(config)
    
    devops_engineer = TeamMember(
        "devops-engineer",
        "Senior DevOps Engineer", 
        rag_context
    )
    devops_engineer.setup_llm(config)
    
    # Create team campfire
    team_campfire = Campfire("development-team")
    team_campfire.add_camper(backend_engineer)
    team_campfire.add_camper(devops_engineer)
    
    await team_campfire.start()
    
    # Ask for team input on a technical decision
    question = Torch(claim="How should we implement user authentication for the new tax module?")
    await team_campfire.send_torch(question)
    
    await team_campfire.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Enhanced Orchestration

The Enhanced Orchestration system provides sophisticated task management with detailed execution tracking. Here's how to leverage these powerful features:

### Basic Enhanced Orchestration Setup

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig
from campfires.core.enhanced_orchestration import EnhancedOrchestration

class AnalystCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str):
        super().__init__(name)
        self.expertise = expertise
        
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Enhanced prompt processing with detailed execution tracking"""
        try:
            enhanced_prompt = f"""
            As a {self.expertise} expert, analyze the following:
            {raw_prompt}
            
            Provide detailed insights including:
            1. Problem understanding
            2. Approach selection
            3. Key considerations
            4. Recommended actions
            """
            
            response = await self.llm_completion(enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "expertise": self.expertise,
                    "analysis_depth": "comprehensive",
                    "execution_stage": "expert_analysis"
                }
            }
        except Exception as e:
            return {
                "claim": f"Analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True}
            }

async def run_enhanced_orchestration():
    # Setup LLM configuration
    config = OpenRouterConfig(api_key="your-api-key")
    
    # Create specialized campers
    business_analyst = AnalystCamper("business-analyst", "business strategy")
    business_analyst.setup_llm(config)
    
    tech_analyst = AnalystCamper("tech-analyst", "technology architecture")
    tech_analyst.setup_llm(config)
    
    # Create campfire with enhanced orchestration
    campfire = Campfire("strategic-analysis")
    campfire.add_camper(business_analyst)
    campfire.add_camper(tech_analyst)
    
    # The enhanced orchestration automatically captures:
    # - Detailed execution stages
    # - Problem understanding phases
    # - Approach selection reasoning
    # - Quality considerations
    # - Risk assessments
    
    await campfire.start()
    
    # Process a complex business question
    from campfires import Torch
    question = Torch(
        claim="Should we migrate our legacy system to microservices?",
        metadata={"priority": "high", "stakeholders": ["engineering", "business"]}
    )
    
    await campfire.send_torch(question)
    await campfire.stop()
    
    # Enhanced HTML report will be generated automatically
    print("Check the generated HTML report for detailed execution analysis!")

if __name__ == "__main__":
    asyncio.run(run_enhanced_orchestration())
```

### Understanding the Interactive HTML Reports

The enhanced orchestration system generates rich HTML reports with expandable sections:

#### **Execution Stages Section** 🔍
Click the arrow to expand and see:
- **Problem Understanding**: How campers interpreted the task
- **Approach Selection**: Why specific strategies were chosen
- **Execution Strategy**: Step-by-step implementation details
- **Quality Considerations**: Quality checks and validations performed
- **Risk Assessment**: Potential risks identified and mitigation strategies

#### **RAG Information Section** 📚
Reveals how document context was used:
- **Document Retrieval**: Which documents were accessed
- **Context Integration**: How information was incorporated
- **Relevance Scoring**: Why specific content was prioritized
- **State Management**: How RAG state evolved during processing

#### **Customization Details Section** ⚙️
Shows how campers adapted their responses:
- **Role-Based Adaptations**: How expertise influenced analysis
- **Personality Integration**: How character traits affected responses
- **Context Awareness**: How situational factors were considered

#### **Impact Analysis Section** 📊
Provides outcome assessment:
- **Decision Quality**: Assessment of recommendation strength
- **Confidence Levels**: Reliability indicators for each insight
- **Follow-up Actions**: Suggested next steps
- **Success Metrics**: How to measure implementation success

### Advanced Multi-Camper Orchestration

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig

class SpecializedTeamMember(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, personality: str, concerns: list):
        super().__init__(name)
        self.role = role
        self.personality = personality
        self.concerns = concerns
        
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Role-specific analysis with personality integration"""
        try:
            role_prompt = f"""
            You are a {self.role} with the following personality: {self.personality}
            Your primary concerns are: {', '.join(self.concerns)}
            
            Task: {raw_prompt}
            
            Provide analysis from your unique perspective, considering:
            1. How this aligns with your role responsibilities
            2. What concerns you might have
            3. What opportunities you see
            4. Your recommended approach
            """
            
            response = await self.llm_completion(role_prompt)
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "role": self.role,
                    "personality_influence": self.personality,
                    "key_concerns": self.concerns,
                    "perspective_type": "role_specialized"
                }
            }
        except Exception as e:
            return {
                "claim": f"Unable to provide {self.role} perspective: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "role": self.role}
            }

async def run_team_orchestration():
    config = OpenRouterConfig(api_key="your-api-key")
    
    # Create diverse team members
    team_members = [
        SpecializedTeamMember(
            "sarah-pm", 
            "Project Manager",
            "detail-oriented and deadline-focused",
            ["timeline adherence", "resource allocation", "stakeholder communication"]
        ),
        SpecializedTeamMember(
            "alex-dev",
            "Senior Developer", 
            "pragmatic and quality-focused",
            ["code maintainability", "technical debt", "performance optimization"]
        ),
        SpecializedTeamMember(
            "jordan-ux",
            "UX Designer",
            "user-centric and creative",
            ["user experience", "accessibility", "design consistency"]
        )
    ]
    
    # Setup LLM for each team member
    for member in team_members:
        member.setup_llm(config)
    
    # Create collaborative campfire
    team_campfire = Campfire("product-development-team")
    for member in team_members:
        team_campfire.add_camper(member)
    
    await team_campfire.start()
    
    # Collaborative decision making
    from campfires import Torch
    decision = Torch(
        claim="We need to redesign our mobile app's onboarding flow to improve user retention",
        metadata={
            "urgency": "high",
            "impact": "user_retention",
            "timeline": "6_weeks"
        }
    )
    
    await team_campfire.send_torch(decision)
    await team_campfire.stop()
    
    print("Team collaboration complete! Check the HTML report for detailed insights from each perspective.")

if __name__ == "__main__":
    asyncio.run(run_team_orchestration())
```

### Leveraging RAG Integration

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig

class RAGEnabledCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, domain: str, rag_context: str):
        super().__init__(name)
        self.domain = domain
        self.rag_context = rag_context
        
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """RAG-enhanced analysis with document context"""
        try:
            rag_enhanced_prompt = f"""
            Domain Expertise: {self.domain}
            
            Available Context:
            {self.rag_context}
            
            Question: {raw_prompt}
            
            Using the provided context and your {self.domain} expertise:
            1. Identify relevant information from the context
            2. Apply domain-specific analysis
            3. Provide evidence-based recommendations
            4. Highlight any gaps in available information
            """
            
            response = await self.llm_completion(rag_enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.92,
                "metadata": {
                    "domain": self.domain,
                    "rag_enhanced": True,
                    "context_utilized": True,
                    "evidence_based": True
                }
            }
        except Exception as e:
            return {
                "claim": f"RAG analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "domain": self.domain}
            }

async def run_rag_orchestration():
    config = OpenRouterConfig(api_key="your-api-key")
    
    # Sample RAG context (in practice, this would come from document retrieval)
    financial_context = """
    Company Financial Overview:
    - Q3 Revenue: $2.4M (15% growth)
    - Operating Expenses: $1.8M
    - Cash Flow: Positive $600K
    - Key Investments: R&D (30%), Marketing (25%), Operations (45%)
    - Market Position: Growing market share in fintech sector
    """
    
    # Create RAG-enabled camper
    financial_analyst = RAGEnabledCamper(
        "financial-analyst",
        "financial analysis",
        financial_context
    )
    financial_analyst.setup_llm(config)
    
    # Create campfire
    analysis_campfire = Campfire("financial-analysis")
    analysis_campfire.add_camper(financial_analyst)
    
    await analysis_campfire.start()
    
    # Ask context-aware question
    from campfires import Torch
    question = Torch(
        claim="Should we increase our R&D investment by 50% next quarter?",
        metadata={"analysis_type": "investment_decision", "timeframe": "Q4"}
    )
    
    await analysis_campfire.send_torch(question)
    await analysis_campfire.stop()
    
    print("RAG-enhanced analysis complete! The HTML report shows how document context influenced the decision.")

if __name__ == "__main__":
    asyncio.run(run_rag_orchestration())
```

## Core Concepts

### Torches - The Light of Knowledge
In our valley, **Torches** are glowing vessels that carry information, insights, and data between campers. Each torch illuminates a piece of knowledge with its own confidence level - some burn bright with certainty, others flicker with uncertainty:

```python
from campfires import Torch

torch = Torch(
    claim="The weather is sunny today",
    confidence=0.95,  # How brightly this torch burns
    metadata={"source": "weather_api", "location": "NYC"}
)
```

### Campers - The Valley Inhabitants
**Campers** are the AI agents sitting around your campfire. Each camper has their own expertise and personality. When a torch is passed to them, they examine it, add their insights, and pass along a new torch with their findings:

```python
from campfires import Camper, Torch

class WeatherCamper(Camper):
    async def process(self, torch: Torch) -> Torch:
        # This camper specializes in weather analysis
        return Torch(claim=f"Weather insight: {torch.claim}")
```

### LLMCamperMixin - Bringing Intelligence to Your Campers
The **LLMCamperMixin** gives your campers the ability to think and reason using Large Language Models. When you mix this into your camper class, they gain access to powerful AI capabilities:

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig

class IntelligentCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        # Setup LLM capabilities
        config = OpenRouterConfig(api_key="your-api-key")
        self.setup_llm(config)
    
    async def process(self, torch: Torch) -> Torch:
        # Use LLM to analyze the torch content
        response = await self.llm_completion_with_mcp(
            f"Analyze this: {torch.claim}"
        )
        return Torch(claim=response, confidence=0.9)
    
    def override_prompt(self, torch: Torch) -> dict:
        # Customize how the LLM processes information
        enhanced_prompt = f"As an expert, analyze: {torch.claim}"
        llm_response = self.llm_completion_with_mcp(enhanced_prompt)
        
        return {
            "claim": llm_response,
            "confidence": 0.85,
            "metadata": {"enhanced": True}
        }
```

### Campfires - The Gathering Circles
A **Campfire** is where your campers gather to collaborate. It orchestrates the conversation, ensuring torches are passed in the right order and that every camper gets a chance to contribute their expertise:

```python
from campfires import Campfire

campfire = Campfire("weather-analysis")
campfire.add_camper(weather_camper)
campfire.add_camper(analysis_camper)
# Now they can work together around the fire
```

### Zeitgeist - The Valley's Internet Knowledge
**Zeitgeist** gives your campers the ability to search the internet for current information, opinions, and trends relevant to their roles. Like having a wise oracle at the campfire who can instantly access the collective knowledge of the world:

```python
from campfires import ZeitgeistCamper, LLMCamperMixin

class ResearchCamper(LLMCamperMixin, Camper):
    def __init__(self, name: str, role: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.set_role(role)  # 'academic', 'developer', 'journalist', etc.
        self.enable_zeitgeist()
    
    async def research_topic(self, topic: str):
        # Get current internet knowledge about the topic
        zeitgeist_info = await self.get_zeitgeist(topic)
        role_opinions = await self.get_role_opinions(topic)
        trending_tools = await self.get_trending_tools(topic)
        return {
            'zeitgeist': zeitgeist_info,
            'opinions': role_opinions,
            'tools': trending_tools
        }
```

### Enhanced Orchestration - The Valley's Wisdom
The **Enhanced Orchestration** system provides sophisticated task management with detailed execution stages. When campers work on complex tasks, the system captures their thought processes, approach selection, and quality considerations:

```python
from campfires import Campfire, EnhancedOrchestration

# Create a campfire with enhanced orchestration
campfire = Campfire("strategic-planning")
orchestration = EnhancedOrchestration(campfire)

# The orchestration system automatically captures:
# - Problem understanding and analysis
# - Approach selection and reasoning
# - Execution strategy and implementation
# - Quality considerations and risk assessment
# - RAG context and document integration
# - Final outcomes and impact analysis

# All this information is available in interactive HTML reports
# with expandable sections for detailed exploration
```

### Interactive HTML Reports - Illuminating the Process
The framework generates rich HTML reports that reveal the inner workings of your campfire collaborations. These reports feature expandable sections that show:

- **Execution Stages**: Step-by-step breakdown of how tasks were approached and executed
- **RAG Information**: Details about document retrieval and context integration
- **Customization**: How campers adapted their responses based on their roles and expertise
- **Impact Analysis**: Assessment of outcomes and recommendations for future improvements

Click the arrow icons in the report headers to expand sections and explore the detailed execution process.

### Party Box - The Valley's Treasure Chest
The **Party Box** is the shared storage system where campfires can exchange valuable artifacts - documents, images, audio files, and data. It's like a magical chest that connects all campfires in the valley:

```python
from campfires import LocalDriver

# Store something in the party box
party_box = LocalDriver("./demo_storage")
await party_box.store_asset(file_data, "shared_document.pdf")
```

### MCP Protocol - The Torch Bearer Network
The **Model Context Protocol** is how torch bearers carry messages between campfires throughout the valley. It ensures that important information, events, and notifications reach every campfire that needs to know:

```python
from campfires import MCPProtocol, AsyncQueueTransport

transport = AsyncQueueTransport()
mcp_protocol = MCPProtocol(transport)
await mcp_protocol.start()
# Now torch bearers can carry messages across the valley
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_DEFAULT_MODEL=anthropic/claude-3-sonnet
CAMPFIRES_LOG_LEVEL=INFO
CAMPFIRES_DB_PATH=./campfires.db
```

### OpenRouter Configuration

```python
from campfires import OpenRouterConfig

config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="anthropic/claude-3-sonnet",
    max_tokens=1000,
    temperature=0.7
)
```

## Examples

Check out the `demos/` directory for complete examples:

- `sequential_orchestration_demo.py`: Advanced task orchestration with detailed execution stages and interactive HTML reports
- `hospital_zeitgeist_demo.py`: Healthcare team collaboration with professional AI personas, action planning, and enhanced reporting
- `tax_app_team_demo.py`: Software development team collaboration with RAG integration, LLM-powered recommendations, and detailed execution analysis
- `zeitgeist_demo.py`: Internet knowledge and opinion mining with Zeitgeist
- `reddit_crisis_tracker.py`: Crisis detection system for social media
- `run_demo.py`: Simple demonstration of basic concepts

All demos generate interactive HTML reports with expandable sections showing execution stages, RAG information, and detailed analysis.

## Development

### Setting up for Development

```bash
git clone https://github.com/campfires/campfires.git
cd campfires
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black campfires/
```

### Type Checking

```bash
mypy campfires/
```

## Optional Dependencies

### Zeitgeist Support
```bash
pip install duckduckgo-search beautifulsoup4 requests
```

### AWS Support
```bash
pip install "campfires[aws]"
```

### Redis Support
```bash
pip install "campfires[redis]"
```

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://campfires.readthedocs.io
- GitHub Issues: https://github.com/campfires/campfires/issues
- Discussions: https://github.com/campfires/campfires/discussions