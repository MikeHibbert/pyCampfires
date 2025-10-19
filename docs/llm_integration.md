# LLM Integration Guide

This guide covers advanced LLM integration patterns in the Campfires framework, including the `LLMCamperMixin`, `TeamMember` class, custom prompt engineering techniques, and the new **Enhanced Orchestration** system.

## Table of Contents

- [Enhanced Orchestration System](#enhanced-orchestration-system) ⭐ **NEW**
- [Interactive HTML Reports](#interactive-html-reports) ⭐ **NEW**
- [Advanced Execution Tracking](#advanced-execution-tracking) ⭐ **NEW**
- [LLMCamperMixin Overview](#llmcampermixin-overview)
- [TeamMember Class with RAG Integration](#teammember-class-with-rag-integration)
- [Custom Prompt Engineering](#custom-prompt-engineering)
- [OpenRouter Configuration](#openrouter-configuration)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Enhanced Orchestration System

The Enhanced Orchestration system represents a major advancement in the Campfires framework, providing sophisticated task management with detailed execution tracking and interactive HTML reports. This system automatically captures the thought processes, decision-making strategies, and quality considerations of your AI agents.

### Key Features

- **Detailed Execution Tracking**: Captures problem understanding, approach selection, and quality considerations
- **Interactive HTML Reports**: Rich reports with expandable sections for deep analysis
- **RAG Integration Tracking**: Shows how document context influences decisions
- **Role-Based Analysis**: Tracks how different expertise areas contribute to outcomes
- **Risk Assessment**: Automatic identification of potential risks and mitigation strategies
- **Quality Validation**: Built-in quality checks and confidence assessments
- **Multi-Stage Processing**: Support for complex workflows with stage-by-stage tracking

### Enhanced LLM Integration

The Enhanced Orchestration system seamlessly integrates with existing LLM capabilities while adding sophisticated tracking and reporting:

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch

class EnhancedLLMCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str, personality: str):
        super().__init__(name)
        self.expertise = expertise
        self.personality = personality
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Enhanced LLM processing with detailed execution tracking"""
        try:
            # The enhanced orchestration system automatically captures:
            # - Problem understanding phase
            # - Approach selection reasoning
            # - Quality considerations
            # - Risk assessments
            # - RAG context utilization
            
            enhanced_prompt = f"""
            You are a {self.expertise} expert with the following personality: {self.personality}
            
            Task: {raw_prompt}
            
            Please provide a comprehensive analysis including:
            
            1. PROBLEM UNDERSTANDING:
               - How you interpret this task
               - Key assumptions you're making
               - Critical factors to consider
               - Stakeholder perspectives
            
            2. APPROACH SELECTION:
               - Your chosen strategy and why
               - Alternative approaches considered
               - Risk-benefit analysis
               - Resource requirements
            
            3. DETAILED ANALYSIS:
               - Step-by-step reasoning
               - Evidence and supporting information
               - Quality checks performed
               - Validation criteria
            
            4. RISK ASSESSMENT:
               - Potential risks identified
               - Impact and probability assessment
               - Mitigation strategies
               - Contingency plans
            
            5. QUALITY CONSIDERATIONS:
               - Accuracy and reliability factors
               - Confidence level and limitations
               - Validation methods used
               - Success metrics
            
            6. RECOMMENDATIONS:
               - Specific actionable recommendations
               - Implementation considerations
               - Timeline and milestones
               - Success criteria
            """
            
            response = await self.llm_completion(enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "expertise": self.expertise,
                    "personality": self.personality,
                    "analysis_depth": "comprehensive",
                    "execution_stage": "enhanced_analysis",
                    "quality_assured": True,
                    "risk_assessed": True,
                    "enhanced_orchestration": True
                }
            }
        except Exception as e:
            return {
                "claim": f"Enhanced analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {
                    "error": True, 
                    "expertise": self.expertise,
                    "enhanced_orchestration": True
                }
            }
```

## Interactive HTML Reports

The Enhanced Orchestration system generates rich, interactive HTML reports that provide deep insights into the decision-making processes of your AI agents. These reports are automatically created when using enhanced orchestration features.

### Report Structure and Features

#### 🔍 Execution Stages
Each report includes detailed tracking of execution stages:

- **Problem Understanding**: How agents interpreted the task
  - Initial problem analysis
  - Assumption identification
  - Stakeholder consideration
  - Context evaluation

- **Approach Selection**: Strategy selection and reasoning
  - Methodology selection rationale
  - Alternative approaches considered
  - Risk-benefit analysis
  - Resource requirement assessment

- **Execution Strategy**: Implementation details and planning
  - Step-by-step execution plan
  - Quality checkpoints
  - Validation criteria
  - Success metrics

- **Quality Considerations**: Quality checks and validations
  - Accuracy assessments
  - Reliability factors
  - Confidence intervals
  - Limitation acknowledgments

- **Risk Assessment**: Risk identification and mitigation
  - Risk identification process
  - Impact and probability analysis
  - Mitigation strategies
  - Contingency planning

#### 📚 RAG Information Tracking
When RAG (Retrieval-Augmented Generation) is used, reports include:

- **Document Retrieval**: Sources accessed and retrieval methods
- **Context Integration**: How information was synthesized
- **Relevance Scoring**: Content prioritization rationale
- **State Management**: Context evolution during processing

#### ⚙️ Customization Details
Reports show how agent characteristics influenced the analysis:

- **Role-Based Adaptations**: How expertise influenced analysis
- **Personality Integration**: Character trait effects on responses
- **Context Awareness**: Situational factor considerations

#### 📊 Impact Analysis
Comprehensive analysis of outcomes and implications:

- **Decision Quality**: Recommendation strength assessment
- **Confidence Levels**: Reliability indicators
- **Follow-up Actions**: Next steps and implementation guidance
- **Success Metrics**: Measurement criteria and benchmarks

### Accessing and Using Reports

```python
# Reports are automatically generated in the project directory
# Look for files named: {session_name}_report_{timestamp}.html

# Example report features:
# 1. Open the HTML file in your browser
# 2. Click the arrow (▶) next to each section to expand
# 3. Explore the detailed execution tracking information
# 4. Review the metadata and decision rationale
# 5. Use the color-coded sections for easy navigation
# 6. Export sections for documentation or sharing
```

## Advanced Execution Tracking

The Enhanced Orchestration system provides sophisticated execution tracking that goes beyond simple input/output logging.

### Multi-Stage Processing

Track complex workflows through multiple processing stages:

```python
class StageProcessor(Camper, LLMCamperMixin):
    def __init__(self, name: str, stage: str, focus_area: str):
        super().__init__(name)
        self.stage = stage
        self.focus_area = focus_area
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Stage-specific processing with enhanced tracking"""
        try:
            # Enhanced orchestration automatically tracks:
            # - Stage progression
            # - Decision points
            # - Quality gates
            # - Risk checkpoints
            
            stage_prompt = f"""
            PROCESSING STAGE: {self.stage}
            FOCUS AREA: {self.focus_area}
            
            Task: {raw_prompt}
            
            As a {self.stage} specialist focusing on {self.focus_area}:
            
            1. STAGE-SPECIFIC ANALYSIS:
               - Key considerations for this stage
               - Critical success factors
               - Quality requirements
               - Risk factors specific to this stage
            
            2. DECISION RATIONALE:
               - Why this approach for this stage
               - Alternative options considered
               - Trade-offs and implications
               - Dependencies on other stages
            
            3. QUALITY ASSURANCE:
               - Validation criteria for this stage
               - Quality checkpoints
               - Success metrics
               - Error detection methods
            
            4. STAGE COMPLETION:
               - Deliverables for this stage
               - Readiness for next stage
               - Handoff requirements
               - Documentation needs
            """
            
            response = await self.llm_completion(stage_prompt)
            
            return {
                "claim": response,
                "confidence": 0.88,
                "metadata": {
                    "processing_stage": self.stage,
                    "focus_area": self.focus_area,
                    "stage_complete": True,
                    "next_stage_ready": True,
                    "quality_validated": True,
                    "enhanced_tracking": True
                }
            }
        except Exception as e:
            return {
                "claim": f"Stage {self.stage} processing failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {
                    "error": True, 
                    "stage": self.stage,
                    "enhanced_tracking": True
                }
            }
```

### RAG-Enhanced Execution Tracking

Combine document context with execution tracking for intelligent analysis:

```python
class RAGEnhancedCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, expertise: str, rag_context: str):
        super().__init__(name)
        self.role = role
        self.expertise = expertise
        self.rag_context = rag_context
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """RAG-enhanced processing with detailed execution tracking"""
        try:
            # Enhanced orchestration tracks RAG utilization:
            # - Document retrieval process
            # - Context integration methods
            # - Relevance scoring
            # - Information synthesis
            
            rag_enhanced_prompt = f"""
            ROLE: {self.role}
            EXPERTISE: {self.expertise}
            
            AVAILABLE CONTEXT DOCUMENTS:
            {self.rag_context}
            
            TASK: {raw_prompt}
            
            As a {self.role} with expertise in {self.expertise}:
            
            1. CONTEXT ANALYSIS:
               - Relevant information from available documents
               - How context applies to this specific task
               - Gaps in available information
               - Additional context needed
            
            2. ROLE-SPECIFIC PERSPECTIVE:
               - Analysis from your professional viewpoint
               - Key concerns and considerations for your role
               - Opportunities and recommendations
               - Professional standards and best practices
            
            3. EVIDENCE-BASED RECOMMENDATIONS:
               - Recommendations supported by context documents
               - Best practices from your expertise area
               - Implementation guidance
               - Risk mitigation strategies
            
            4. COLLABORATION INSIGHTS:
               - How this integrates with other team perspectives
               - Dependencies and coordination requirements
               - Success criteria from your role's perspective
               - Communication and handoff requirements
            
            5. QUALITY AND VALIDATION:
               - Confidence in recommendations
               - Validation methods used
               - Limitations and assumptions
               - Success metrics and monitoring
            """
            
            response = await self.llm_completion(rag_enhanced_prompt)
            
            return {
                "claim": response,
                "confidence": 0.92,
                "metadata": {
                    "role": self.role,
                    "expertise": self.expertise,
                    "rag_enhanced": True,
                    "context_utilized": True,
                    "evidence_based": True,
                    "collaboration_ready": True,
                    "quality_validated": True,
                    "enhanced_orchestration": True
                }
            }
        except Exception as e:
            return {
                "claim": f"RAG analysis failed for {self.role}: {str(e)}",
                "confidence": 0.1,
                "metadata": {
                    "error": True, 
                    "role": self.role,
                    "enhanced_orchestration": True
                }
            }
```

## LLMCamperMixin Overview

The `LLMCamperMixin` provides Large Language Model capabilities to any Camper class. It handles LLM client setup, API calls, and provides convenient methods for interacting with language models.

### Key Features

- **OpenRouter Integration**: Built-in support for OpenRouter API with multiple model providers
- **MCP Protocol Support**: Model Context Protocol for inter-agent communication
- **Flexible Configuration**: Support for different models, parameters, and providers
- **Error Handling**: Robust error handling for API failures and network issues

### Basic Usage

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig

class IntelligentCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        
        # Setup LLM configuration
        config = OpenRouterConfig(api_key="your-openrouter-api-key")
        self.setup_llm(config)
    
    async def process(self, torch: Torch) -> Torch:
        # Use LLM for processing
        response = await self.llm_completion_with_mcp(
            f"Analyze this content: {torch.claim}"
        )
        
        return Torch(
            claim=response,
            confidence=0.9,
            metadata={"processed_by": "llm"}
        )
```

### Available Methods

- `setup_llm(config, mcp_protocol=None)`: Initialize LLM client with configuration
- `llm_completion_with_mcp(prompt, **kwargs)`: Make LLM completion call with MCP support
- `llm_chat_with_mcp(messages, **kwargs)`: Make LLM chat call with message history

## TeamMember Class with RAG Integration

The `TeamMember` class demonstrates how to build intelligent agents that combine role-based expertise with document context through RAG (Retrieval-Augmented Generation).

### Class Definition

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig

class TeamMember(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, rag_system_prompt: str):
        super().__init__(name)
        self.role = role
        self.rag_system_prompt = rag_system_prompt
        
        # Setup LLM configuration
        config = OpenRouterConfig()  # Uses environment variables
        self.setup_llm(config)
    
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
```

### Key Features

- **Role-Based Expertise**: Each team member has a specific professional role
- **RAG Integration**: Access to comprehensive documentation and context
- **Custom Prompt Engineering**: Uses `override_prompt` for sophisticated LLM interactions
- **Error Handling**: Graceful degradation when LLM calls fail
- **Metadata Tracking**: Rich metadata for response analysis

### Usage Example

```python
import asyncio
from campfires import Campfire

async def main():
    # RAG system prompt with comprehensive context
    rag_context = """
    You have access to comprehensive documentation about our tax application system.
    The system handles tax calculations, user management, and compliance reporting.
    Key components include:
    - Authentication service with OAuth2 and JWT tokens
    - Calculation engine with support for federal and state taxes
    - Reporting module with PDF generation and email delivery
    - Database layer with PostgreSQL and Redis caching
    - API gateway with rate limiting and monitoring
    """
    
    # Create team members with different roles
    backend_engineer = TeamMember(
        "backend-engineer", 
        "Senior Backend Engineer",
        rag_context
    )
    
    devops_engineer = TeamMember(
        "devops-engineer",
        "Senior DevOps Engineer", 
        rag_context
    )
    
    testing_engineer = TeamMember(
        "testing-engineer",
        "Senior Testing Engineer",
        rag_context
    )
    
    # Create team campfire
    team_campfire = Campfire("development-team")
    team_campfire.add_camper(backend_engineer)
    team_campfire.add_camper(devops_engineer)
    team_campfire.add_camper(testing_engineer)
    
    await team_campfire.start()
    
    # Ask for team input on technical decisions
    questions = [
        "How should we implement user authentication for the new tax module?",
        "What's the best approach for handling high-volume tax calculations?",
        "How can we ensure data security and compliance in our system?"
    ]
    
    for question in questions:
        torch = Torch(claim=question)
        await team_campfire.send_torch(torch)
    
    await team_campfire.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Custom Prompt Engineering

The `override_prompt` method allows for sophisticated prompt engineering and custom LLM interactions.

### Method Signature

```python
def override_prompt(self, torch: Torch) -> dict:
    """
    Override the default prompt processing with custom LLM interactions.
    
    Args:
        torch: The input torch containing the claim to process
        
    Returns:
        dict: Dictionary with 'claim', 'confidence', and 'metadata' keys
    """
```

### Advanced Example

```python
class ExpertAnalyzer(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str, context_docs: list):
        super().__init__(name)
        self.expertise = expertise
        self.context_docs = context_docs
        
    def override_prompt(self, torch: Torch) -> dict:
        """Advanced prompt engineering with multi-step reasoning"""
        try:
            # Step 1: Context retrieval
            relevant_context = self._retrieve_relevant_context(torch.claim)
            
            # Step 2: Expert analysis prompt
            analysis_prompt = f"""
            You are a world-class expert in {self.expertise}.
            
            Context Documents:
            {relevant_context}
            
            Analysis Request: {torch.claim}
            
            Please provide:
            1. Executive Summary (2-3 sentences)
            2. Detailed Analysis (key insights and implications)
            3. Risk Assessment (potential concerns and mitigation strategies)
            4. Recommendations (specific, actionable next steps)
            5. Confidence Level (your confidence in this analysis, 0-100%)
            
            Format your response as structured sections.
            """
            
            # Step 3: LLM call with structured prompt
            response = self.llm_completion_with_mcp(
                analysis_prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Step 4: Extract confidence from response
            confidence = self._extract_confidence(response)
            
            return {
                "claim": response,
                "confidence": confidence / 100.0,  # Convert to 0-1 scale
                "metadata": {
                    "expertise": self.expertise,
                    "analysis_type": "expert_structured",
                    "context_docs_used": len(relevant_context),
                    "prompt_engineering": "multi_step"
                }
            }
            
        except Exception as e:
            return {
                "claim": f"Expert analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {
                    "error": True,
                    "expertise": self.expertise,
                    "error_type": type(e).__name__
                }
            }
    
    def _retrieve_relevant_context(self, query: str) -> str:
        """Simulate document retrieval for RAG"""
        # In a real implementation, this would use vector search
        # or other retrieval mechanisms
        return "\n".join(self.context_docs[:3])  # Top 3 relevant docs
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from LLM response"""
        # Simple regex to find confidence percentage
        import re
        match = re.search(r'confidence.*?(\d+)%', response.lower())
        return float(match.group(1)) if match else 75.0  # Default confidence
```

## OpenRouter Configuration

The framework uses OpenRouter for LLM API access, supporting multiple model providers.

### Environment Variables

Create a `.env` file in your project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_DEFAULT_MODEL=anthropic/claude-3-5-sonnet-20241022
OPENROUTER_MAX_TOKENS=2000
OPENROUTER_TEMPERATURE=0.7
```

### Programmatic Configuration

```python
from campfires import OpenRouterConfig

# Basic configuration
config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="anthropic/claude-3-5-sonnet-20241022"
)

# Advanced configuration
config = OpenRouterConfig(
    api_key="your-api-key",
    default_model="anthropic/claude-3-5-sonnet-20241022",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

### Supported Models

OpenRouter supports models from various providers:

- **Anthropic**: `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-haiku`
- **OpenAI**: `openai/gpt-4-turbo`, `openai/gpt-3.5-turbo`
- **Google**: `google/gemini-pro`, `google/gemini-pro-vision`
- **Meta**: `meta-llama/llama-2-70b-chat`, `meta-llama/codellama-34b-instruct`

## Best Practices

### 1. Prompt Design

- **Be Specific**: Clearly define the task and expected output format
- **Provide Context**: Include relevant background information and constraints
- **Use Examples**: Show the model what good responses look like
- **Structure Requests**: Break complex tasks into clear steps

### 2. Error Handling

- **Always Use Try-Catch**: Wrap LLM calls in exception handling
- **Provide Fallbacks**: Return meaningful responses when LLM calls fail
- **Log Errors**: Track API failures for debugging and monitoring
- **Graceful Degradation**: Maintain functionality even when LLM is unavailable

### 3. Performance Optimization

- **Cache Responses**: Store frequently requested analyses
- **Batch Requests**: Group similar requests when possible
- **Monitor Usage**: Track API costs and rate limits
- **Choose Appropriate Models**: Balance cost, speed, and quality

### 4. Security Considerations

- **Protect API Keys**: Use environment variables, never hardcode keys
- **Validate Inputs**: Sanitize user inputs before sending to LLM
- **Filter Outputs**: Review LLM responses for sensitive information
- **Rate Limiting**: Implement appropriate rate limiting for your use case

## Error Handling

### Common Error Scenarios

```python
def override_prompt(self, torch: Torch) -> dict:
    """Robust error handling for LLM interactions"""
    try:
        response = self.llm_completion_with_mcp(prompt)
        return {
            "claim": response,
            "confidence": 0.9,
            "metadata": {"success": True}
        }
    
    except APIConnectionError as e:
        # Network connectivity issues
        return {
            "claim": "Unable to connect to LLM service. Please try again later.",
            "confidence": 0.0,
            "metadata": {"error": "connection_error", "details": str(e)}
        }
    
    except RateLimitError as e:
        # API rate limit exceeded
        return {
            "claim": "Service temporarily unavailable due to high demand.",
            "confidence": 0.0,
            "metadata": {"error": "rate_limit", "retry_after": e.retry_after}
        }
    
    except AuthenticationError as e:
        # Invalid API key or authentication failure
        return {
            "claim": "Authentication failed. Please check API configuration.",
            "confidence": 0.0,
            "metadata": {"error": "auth_error", "details": str(e)}
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        return {
            "claim": f"Unexpected error occurred: {str(e)}",
            "confidence": 0.0,
            "metadata": {"error": "unknown", "type": type(e).__name__}
        }
```

## Examples

### Complete Tax Application Team Demo

See `demos/tax_app_team_demo.py` for a comprehensive example that demonstrates:

- Multiple team members with different roles
- RAG integration with system documentation
- Custom prompt engineering for each role
- Error handling and graceful degradation
- HTML report generation with team recommendations

### Expert Analysis System

```python
# Create expert analyzers for different domains
security_expert = ExpertAnalyzer("security", "cybersecurity", security_docs)
finance_expert = ExpertAnalyzer("finance", "financial analysis", finance_docs)
legal_expert = ExpertAnalyzer("legal", "regulatory compliance", legal_docs)

# Create multi-expert campfire
expert_campfire = Campfire("expert-panel")
expert_campfire.add_camper(security_expert)
expert_campfire.add_camper(finance_expert)
expert_campfire.add_camper(legal_expert)

# Analyze complex business decisions
await expert_campfire.send_torch(
    Torch(claim="Should we implement blockchain-based payments?")
)
```

### Crisis Response Team

```python
# Create crisis response specialists
crisis_detector = CrisisDetectionCamper("crisis-detector")
response_generator = ResponseGeneratorCamper("response-generator")
escalation_manager = EscalationManagerCamper("escalation-manager")

# Create crisis response campfire
crisis_campfire = Campfire("crisis-response")
crisis_campfire.add_camper(crisis_detector)
crisis_campfire.add_camper(response_generator)
crisis_campfire.add_camper(escalation_manager)

# Process incoming crisis reports
await crisis_campfire.send_torch(
    Torch(claim="User reported feeling overwhelmed and hopeless")
)
```

## Conclusion

The LLM integration capabilities in Campfires provide powerful tools for building intelligent, collaborative AI systems. By combining the `LLMCamperMixin`, custom prompt engineering, and RAG integration, you can create sophisticated agents that provide expert-level insights and recommendations.

Key takeaways:

1. Use `LLMCamperMixin` for basic LLM capabilities
2. Implement `override_prompt` for custom prompt engineering
3. Combine RAG with role-based expertise for intelligent team members
4. Always implement robust error handling
5. Follow security best practices for API key management
6. Monitor performance and costs in production environments

For more examples and advanced patterns, explore the demos in the `demos/` directory.