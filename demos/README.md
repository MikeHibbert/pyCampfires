# Campfires Framework Demos

This directory contains demonstration scripts that showcase the capabilities of the Campfires framework, including the new **Enhanced Orchestration** system with interactive HTML reports.

## 🚀 New Enhanced Orchestration Features

All demos now include **Enhanced Orchestration** capabilities that provide:

### 📊 Interactive HTML Reports
- **Expandable Execution Stages**: Click to see detailed problem understanding, approach selection, and execution strategies
- **RAG Information Tracking**: View how document context influences decisions
- **Customization Details**: See how camper personalities and roles affect responses
- **Impact Analysis**: Assess decision quality, confidence levels, and follow-up actions

### 🔍 Detailed Execution Tracking
- **Problem Understanding**: How each camper interprets the task
- **Approach Selection**: Why specific strategies were chosen
- **Quality Considerations**: Quality checks and validations performed
- **Risk Assessment**: Potential risks and mitigation strategies

### 🎯 Enhanced Team Collaboration
- **Role-Based Analysis**: Each team member provides perspective based on their expertise
- **Personality Integration**: Character traits influence response styles
- **Context Awareness**: Situational factors are considered in recommendations

## Available Demos

### 1. Sequential Orchestration Demo (`sequential_orchestration_demo.py`) ⭐ NEW

A comprehensive demonstration of the Enhanced Orchestration system featuring sequential task processing with detailed execution tracking.

**Enhanced Orchestration Features:**
- **Multi-Stage Processing**: Tasks flow through multiple specialized campers in sequence
- **Execution Stage Tracking**: Detailed capture of problem understanding, approach selection, and quality considerations
- **Interactive HTML Reports**: Rich reports with expandable sections for execution stages, RAG information, customization details, and impact analysis
- **Risk Assessment**: Automatic identification of potential risks and mitigation strategies
- **Quality Validation**: Built-in quality checks and confidence scoring

**Technical Implementation:**
- **Enhanced Orchestration Engine**: Demonstrates the new orchestration system with detailed tracking
- **Sequential Processing**: Shows how tasks flow through multiple processing stages
- **Metadata Enrichment**: Each processing stage adds rich metadata for analysis
- **Report Generation**: Automatic generation of interactive HTML reports

**Processing Stages:**
1. **Initial Analysis**: Problem understanding and approach selection
2. **Detailed Processing**: In-depth analysis with quality considerations
3. **Risk Assessment**: Identification of potential risks and mitigation strategies
4. **Final Validation**: Quality checks and confidence scoring

**To run:**
```bash
python demos/sequential_orchestration_demo.py
```

**Output:** Generates interactive HTML reports showing detailed execution stages, decision-making processes, and quality assessments.

### 2. Tax Application Team Demo (`tax_app_team_demo.py`)

A comprehensive software development team collaboration simulator that demonstrates advanced LLM integration with RAG (Retrieval-Augmented Generation) capabilities and Enhanced Orchestration.

**Enhanced Orchestration Features:**
- **Interactive HTML Reports**: Rich reports with expandable sections showing how each team member's expertise influences decisions
- **Execution Stage Tracking**: Detailed capture of problem understanding, approach selection, and technical considerations
- **RAG Information Display**: Visual representation of how documentation context influences team recommendations
- **Role-Based Analysis**: Each team member's perspective is tracked and analyzed for decision quality
- **Impact Assessment**: Evaluation of recommendation strength and implementation feasibility

**Core Features:**
- **LLM-Powered Team Members**: Senior Backend Engineer, DevOps Engineer, Testing Engineer, and Frontend Developer
- **RAG Integration**: Team members have access to comprehensive documentation about the tax application system
- **Real LLM Responses**: Uses OpenRouter API with models like Claude-3.5-Sonnet for intelligent recommendations
- **Custom Prompt Engineering**: Implements `override_prompt` method for sophisticated LLM interactions
- **Professional Expertise**: Each team member provides role-specific insights and recommendations
- **Enhanced HTML Reports**: Creates detailed meeting reports with actionable recommendations and execution analysis

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

### 3. Hospital Zeitgeist Demo (`hospital_zeitgeist_demo.py`)

A sophisticated healthcare team collaboration simulator that demonstrates advanced multi-agent conversations with professional AI personas and Enhanced Orchestration.

**Enhanced Orchestration Features:**
- **Healthcare-Specific Execution Tracking**: Detailed capture of clinical reasoning, policy considerations, and patient safety assessments
- **Interactive Healthcare Reports**: Rich HTML reports with expandable sections for clinical decision-making processes
- **Professional Persona Analysis**: How each healthcare role's expertise and concerns influence recommendations
- **Risk Assessment for Healthcare**: Identification of patient safety risks, compliance issues, and operational challenges
- **Evidence-Based Decision Tracking**: How Zeitgeist research influences healthcare policy and procedure recommendations

**Core Features:**
- **Professional Healthcare Characters**: Head Nurse, Admin Coordinator, Patient Advocate, IT Specialist, and Ward Manager
- **Zeitgeist Integration**: Real-time internet research for informed healthcare discussions
- **Action Planning**: Generates structured action plans with priorities, timelines, and responsible parties
- **Enhanced HTML Reporting**: Creates detailed meeting reports with character responses, action items, and execution analysis
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

### 6. Multimodal Capabilities Demo (`multimodal_demo.py`) ⭐ NEW

A comprehensive demonstration of Campfires' advanced multimodal content handling capabilities, showcasing the framework's ability to process text, images, audio, and other content types in a unified system.

**Multimodal Features:**
- **Content Type Management**: Unified handling of text, image, audio, video, and document content
- **Multimodal Torches**: Enhanced torch system supporting mixed content types
- **Asset Management**: Advanced storage and retrieval with automatic metadata extraction
- **Prompt Engineering**: Specialized prompts for different content types and multimodal scenarios
- **Audio Processing**: Format detection, validation, conversion, and metadata extraction
- **OpenRouter Integration**: Multimodal API integration for vision and audio analysis
- **Search and Indexing**: Content-aware search across different media types

**Technical Implementation:**
- **MultimodalContent**: Core class for handling different content types with metadata
- **MultimodalTorch**: Enhanced torch supporting mixed content with type filtering
- **MultimodalAssetManager**: Advanced storage with content classification and search
- **MultimodalPromptLibrary**: Template-based prompts for multimodal scenarios
- **AudioProcessor**: Comprehensive audio format handling and processing
- **MetadataExtractor**: Automatic metadata extraction for all content types

**Processing Capabilities:**
1. **Content Creation**: Create and manage text, image, audio, and document content
2. **Metadata Extraction**: Automatic extraction of content properties and characteristics
3. **Asset Storage**: Intelligent storage with deduplication and content-based organization
4. **Search and Discovery**: Advanced search by content type, metadata, and properties
5. **Format Conversion**: Convert between different content formats and encodings
6. **Prompt Engineering**: Generate specialized prompts for different content analysis tasks

**To run:**
```bash
python demos/multimodal_demo.py
```

**Output:** Creates sample multimodal assets, demonstrates processing capabilities, and generates a comprehensive demo report with statistics and examples.

### 7. Quick Multimodal Demo (`quick_multimodal_demo.py`) ⭐ NEW

A streamlined demonstration of core multimodal features for quick exploration and testing.

**Core Features:**
- **Multimodal Content Creation**: Simple examples of text, image, and audio content handling
- **Content Operations**: Filtering, analysis, and format conversion
- **Basic Prompt Engineering**: Essential multimodal prompt generation
- **Integration Examples**: Code examples for OpenRouter, Campfire, and Party Box integration

**Quick Start Features:**
- **ContentType Enumeration**: Understanding different content types
- **MultimodalContent Operations**: Creating and manipulating content objects
- **MultimodalTorch Basics**: Essential torch operations for multimodal content
- **Format Conversion**: Converting between MCP messages and legacy formats

**To run:**
```bash
python demos/quick_multimodal_demo.py
```

**Output:** Provides immediate feedback on multimodal capabilities with concise examples and integration patterns.

## Understanding Enhanced Orchestration Output

### Interactive HTML Reports Structure

All enhanced orchestration demos generate rich HTML reports with the following expandable sections:

#### 🔍 **Execution Stages Section**
Click the arrow to expand and explore:

**Problem Understanding**
- How each camper interpreted the task or question
- What assumptions were made
- How context influenced understanding
- Key factors identified for consideration

**Approach Selection**
- Why specific strategies were chosen
- Alternative approaches considered
- Decision-making rationale
- Risk-benefit analysis of chosen approach

**Execution Strategy**
- Step-by-step implementation details
- Resource requirements identified
- Timeline considerations
- Quality checkpoints planned

**Quality Considerations**
- Quality checks and validations performed
- Standards and criteria applied
- Verification methods used
- Confidence assessment factors

**Risk Assessment**
- Potential risks identified
- Impact and probability analysis
- Mitigation strategies proposed
- Contingency planning considerations

#### 📚 **RAG Information Section**
Reveals how document context influenced decisions:

**Document Retrieval**
- Which documents or knowledge sources were accessed
- Search queries and retrieval methods used
- Relevance scoring and ranking

**Context Integration**
- How retrieved information was incorporated
- Synthesis of multiple sources
- Conflict resolution between sources

**Relevance Scoring**
- Why specific content was prioritized
- Quality assessment of sources
- Confidence in retrieved information

**State Management**
- How RAG state evolved during processing
- Context window management
- Information persistence strategies

#### ⚙️ **Customization Details Section**
Shows how campers adapted their responses:

**Role-Based Adaptations**
- How professional expertise influenced analysis
- Role-specific considerations applied
- Domain knowledge utilization

**Personality Integration**
- How character traits affected response style
- Communication preferences reflected
- Behavioral patterns exhibited

**Context Awareness**
- How situational factors were considered
- Environmental constraints acknowledged
- Stakeholder perspectives integrated

#### 📊 **Impact Analysis Section**
Provides comprehensive outcome assessment:

**Decision Quality**
- Assessment of recommendation strength
- Evidence quality evaluation
- Logical consistency analysis

**Confidence Levels**
- Reliability indicators for each insight
- Uncertainty quantification
- Confidence interval estimation

**Follow-up Actions**
- Suggested next steps and recommendations
- Implementation guidance
- Monitoring and evaluation plans

**Success Metrics**
- How to measure implementation success
- Key performance indicators
- Evaluation criteria and benchmarks

### Reading the Reports

1. **Start with the Summary**: Each report begins with a high-level summary of the session
2. **Explore Execution Stages**: Click to see detailed decision-making processes
3. **Review RAG Integration**: Understand how external knowledge influenced outcomes
4. **Analyze Customizations**: See how roles and personalities shaped responses
5. **Assess Impact**: Evaluate the quality and actionability of recommendations

### Report Navigation Tips

- **Expandable Sections**: Click arrows (▶) to expand detailed information
- **Color Coding**: Different sections use distinct colors for easy navigation
- **Timestamps**: All entries include timestamps for process tracking
- **Metadata**: Rich metadata provides context for each decision point
- **Cross-References**: Links between related sections for comprehensive understanding

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