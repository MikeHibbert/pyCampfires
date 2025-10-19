# LLM Integration Usage Examples

This document provides practical, ready-to-use examples of implementing LLM-enabled Campers using the Campfires framework. Each example demonstrates different patterns and use cases for the `LLMCamperMixin` and `override_prompt` functionality, including the new **Enhanced Orchestration** system.

## Table of Contents

- [Enhanced Orchestration System](#enhanced-orchestration-system) ⭐ **NEW**
- [Interactive HTML Reports](#interactive-html-reports) ⭐ **NEW**
- [Sequential Processing with Enhanced Tracking](#sequential-processing-with-enhanced-tracking) ⭐ **NEW**
- [RAG-Enhanced Team Collaboration](#rag-enhanced-team-collaboration) ⭐ **NEW**
- [Basic LLM-Enabled Camper](#basic-llm-enabled-camper)
- [Expert Analysis System](#expert-analysis-system)
- [Content Generation Pipeline](#content-generation-pipeline)
- [Multi-Agent Collaboration](#multi-agent-collaboration)
- [Crisis Detection and Response](#crisis-detection-and-response)
- [Code Review Assistant](#code-review-assistant)
- [Research and Summarization](#research-and-summarization)
- [Customer Support Bot](#customer-support-bot)
- [Data Analysis Agent](#data-analysis-agent)
- [Creative Writing Assistant](#creative-writing-assistant)

## Enhanced Orchestration System

The Enhanced Orchestration system provides sophisticated task management with detailed execution tracking and interactive HTML reports. This system automatically captures the thought processes, decision-making strategies, and quality considerations of your AI agents.

### Key Features

- **Detailed Execution Tracking**: Captures problem understanding, approach selection, and quality considerations
- **Interactive HTML Reports**: Rich reports with expandable sections for deep analysis
- **RAG Integration Tracking**: Shows how document context influences decisions
- **Role-Based Analysis**: Tracks how different expertise areas contribute to outcomes
- **Risk Assessment**: Automatic identification of potential risks and mitigation strategies

### Basic Enhanced Orchestration Example

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig, Torch

class EnhancedAnalyst(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str, personality: str):
        super().__init__(name)
        self.expertise = expertise
        self.personality = personality
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Enhanced prompt processing with detailed execution tracking"""
        try:
            # The enhanced orchestration system automatically captures:
            # - Problem understanding phase
            # - Approach selection reasoning
            # - Quality considerations
            # - Risk assessments
            
            enhanced_prompt = f"""
            You are a {self.expertise} expert with the following personality: {self.personality}
            
            Task: {raw_prompt}
            
            Please provide a comprehensive analysis including:
            
            1. PROBLEM UNDERSTANDING:
               - How you interpret this task
               - Key assumptions you're making
               - Critical factors to consider
            
            2. APPROACH SELECTION:
               - Your chosen strategy and why
               - Alternative approaches considered
               - Risk-benefit analysis
            
            3. DETAILED ANALYSIS:
               - Step-by-step reasoning
               - Evidence and supporting information
               - Quality checks performed
            
            4. RISK ASSESSMENT:
               - Potential risks identified
               - Mitigation strategies
               - Confidence level and limitations
            
            5. RECOMMENDATIONS:
               - Specific actionable recommendations
               - Implementation considerations
               - Success metrics
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
                    "quality_assured": True
                }
            }
        except Exception as e:
            return {
                "claim": f"Analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "expertise": self.expertise}
            }

async def enhanced_orchestration_example():
    # Setup LLM configuration
    config = OpenRouterConfig()
    
    # Create enhanced analysts with different expertise
    business_analyst = EnhancedAnalyst(
        "business-strategist", 
        "business strategy", 
        "analytical and strategic"
    )
    business_analyst.setup_llm(config)
    
    tech_analyst = EnhancedAnalyst(
        "tech-architect", 
        "technology architecture", 
        "pragmatic and detail-oriented"
    )
    tech_analyst.setup_llm(config)
    
    risk_analyst = EnhancedAnalyst(
        "risk-assessor", 
        "risk management", 
        "cautious and thorough"
    )
    risk_analyst.setup_llm(config)
    
    # Create campfire with enhanced orchestration
    strategic_campfire = Campfire("strategic-analysis")
    strategic_campfire.add_camper(business_analyst)
    strategic_campfire.add_camper(tech_analyst)
    strategic_campfire.add_camper(risk_analyst)
    
    await strategic_campfire.start()
    
    # Process a complex business decision
    decision_torch = Torch(
        claim="Should we migrate our monolithic application to a microservices architecture?",
        metadata={
            "priority": "high",
            "stakeholders": ["engineering", "business", "operations"],
            "timeline": "6_months",
            "budget_impact": "significant"
        }
    )
    
    await strategic_campfire.send_torch(decision_torch)
    await strategic_campfire.stop()
    
    print("Enhanced orchestration complete!")
    print("Check the generated HTML report for detailed execution analysis.")
    print("The report includes expandable sections for:")
    print("- Execution stages and decision-making processes")
    print("- Quality considerations and risk assessments")
    print("- Role-based analysis and expertise contributions")
    print("- Impact analysis and success metrics")

if __name__ == "__main__":
    asyncio.run(enhanced_orchestration_example())
```

## Interactive HTML Reports

The Enhanced Orchestration system generates rich, interactive HTML reports that provide deep insights into the decision-making processes of your AI agents.

### Report Structure

Each HTML report contains the following expandable sections:

#### 🔍 Execution Stages
- **Problem Understanding**: How agents interpreted the task
- **Approach Selection**: Strategy selection and reasoning
- **Execution Strategy**: Implementation details and planning
- **Quality Considerations**: Quality checks and validations
- **Risk Assessment**: Risk identification and mitigation

#### 📚 RAG Information
- **Document Retrieval**: Sources accessed and retrieval methods
- **Context Integration**: How information was synthesized
- **Relevance Scoring**: Content prioritization rationale
- **State Management**: Context evolution during processing

#### ⚙️ Customization Details
- **Role-Based Adaptations**: How expertise influenced analysis
- **Personality Integration**: Character trait effects on responses
- **Context Awareness**: Situational factor considerations

#### 📊 Impact Analysis
- **Decision Quality**: Recommendation strength assessment
- **Confidence Levels**: Reliability indicators
- **Follow-up Actions**: Next steps and implementation guidance
- **Success Metrics**: Measurement criteria and benchmarks

### Accessing Report Features

```python
# Reports are automatically generated in the project directory
# Look for files named: {session_name}_report_{timestamp}.html

# Example report navigation:
# 1. Open the HTML file in your browser
# 2. Click the arrow (▶) next to each section to expand
# 3. Explore the detailed execution tracking information
# 4. Review the metadata and decision rationale
# 5. Use the color-coded sections for easy navigation
```

## Sequential Processing with Enhanced Tracking

Demonstrate how tasks flow through multiple processing stages with detailed tracking.

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig, Torch

class SequentialProcessor(Camper, LLMCamperMixin):
    def __init__(self, name: str, stage: str, focus_area: str):
        super().__init__(name)
        self.stage = stage
        self.focus_area = focus_area
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """Stage-specific processing with enhanced tracking"""
        try:
            stage_prompts = {
                "initial_analysis": f"""
                STAGE: Initial Analysis
                FOCUS: {self.focus_area}
                
                Task: {raw_prompt}
                
                Perform initial analysis focusing on:
                1. Problem decomposition and understanding
                2. Key stakeholders and requirements identification
                3. Initial approach recommendations
                4. Critical success factors
                
                Provide structured analysis with clear reasoning.
                """,
                
                "detailed_processing": f"""
                STAGE: Detailed Processing
                FOCUS: {self.focus_area}
                
                Building on previous analysis: {raw_prompt}
                
                Provide detailed processing including:
                1. In-depth technical/business analysis
                2. Resource requirements and constraints
                3. Implementation roadmap
                4. Quality assurance considerations
                
                Focus on actionable details and practical implementation.
                """,
                
                "risk_assessment": f"""
                STAGE: Risk Assessment
                FOCUS: {self.focus_area}
                
                Analyzing: {raw_prompt}
                
                Conduct comprehensive risk assessment:
                1. Identify potential risks and challenges
                2. Assess impact and probability
                3. Develop mitigation strategies
                4. Create contingency plans
                
                Provide risk matrix and mitigation roadmap.
                """,
                
                "final_validation": f"""
                STAGE: Final Validation
                FOCUS: {self.focus_area}
                
                Final review of: {raw_prompt}
                
                Perform final validation including:
                1. Quality assurance and compliance check
                2. Stakeholder alignment verification
                3. Success metrics definition
                4. Go/no-go recommendation
                
                Provide final recommendation with confidence assessment.
                """
            }
            
            prompt = stage_prompts.get(self.stage, raw_prompt)
            response = await self.llm_completion(prompt)
            
            return {
                "claim": response,
                "confidence": 0.88,
                "metadata": {
                    "processing_stage": self.stage,
                    "focus_area": self.focus_area,
                    "stage_complete": True,
                    "next_stage_ready": True
                }
            }
        except Exception as e:
            return {
                "claim": f"Stage {self.stage} processing failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "stage": self.stage}
            }

async def sequential_processing_example():
    config = OpenRouterConfig()
    
    # Create sequential processing stages
    processors = [
        SequentialProcessor("initial-analyzer", "initial_analysis", "requirements_gathering"),
        SequentialProcessor("detail-processor", "detailed_processing", "technical_implementation"),
        SequentialProcessor("risk-assessor", "risk_assessment", "risk_management"),
        SequentialProcessor("final-validator", "final_validation", "quality_assurance")
    ]
    
    # Setup LLM for each processor
    for processor in processors:
        processor.setup_llm(config)
    
    # Create sequential processing campfire
    sequential_campfire = Campfire("sequential-processing")
    for processor in processors:
        sequential_campfire.add_camper(processor)
    
    await sequential_campfire.start()
    
    # Process through all stages
    project_torch = Torch(
        claim="Develop a new customer onboarding system for our SaaS platform",
        metadata={
            "project_type": "software_development",
            "complexity": "high",
            "timeline": "3_months",
            "team_size": "8_developers"
        }
    )
    
    await sequential_campfire.send_torch(project_torch)
    await sequential_campfire.stop()
    
    print("Sequential processing complete!")
    print("The HTML report shows the complete processing pipeline with:")
    print("- Stage-by-stage execution details")
    print("- Decision flow and reasoning")
    print("- Quality gates and validation points")
    print("- Risk assessment at each stage")

if __name__ == "__main__":
    asyncio.run(sequential_processing_example())
```

## RAG-Enhanced Team Collaboration

Combine document context with team expertise for intelligent collaboration.

```python
import asyncio
from campfires import Campfire, Camper, LLMCamperMixin, OpenRouterConfig, Torch

class RAGTeamMember(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, expertise: str, rag_context: str):
        super().__init__(name)
        self.role = role
        self.expertise = expertise
        self.rag_context = rag_context
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        """RAG-enhanced team collaboration with role-specific expertise"""
        try:
            rag_enhanced_prompt = f"""
            ROLE: {self.role}
            EXPERTISE: {self.expertise}
            
            AVAILABLE CONTEXT DOCUMENTS:
            {self.rag_context}
            
            TASK: {raw_prompt}
            
            As a {self.role} with expertise in {self.expertise}, analyze the task using:
            
            1. CONTEXT ANALYSIS:
               - Relevant information from available documents
               - How context applies to this specific task
               - Gaps in available information
            
            2. ROLE-SPECIFIC PERSPECTIVE:
               - Analysis from your professional viewpoint
               - Key concerns and considerations for your role
               - Opportunities and recommendations
            
            3. EVIDENCE-BASED RECOMMENDATIONS:
               - Recommendations supported by context documents
               - Best practices from your expertise area
               - Implementation guidance
            
            4. COLLABORATION INSIGHTS:
               - How this integrates with other team perspectives
               - Dependencies and coordination requirements
               - Success criteria from your role's perspective
            
            Provide detailed analysis with clear references to context sources.
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
                    "collaboration_ready": True
                }
            }
        except Exception as e:
            return {
                "claim": f"RAG analysis failed for {self.role}: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "role": self.role}
            }

async def rag_team_collaboration_example():
    config = OpenRouterConfig()
    
    # Sample RAG context (in practice, this would come from document retrieval)
    company_context = """
    COMPANY DOCUMENTATION:
    
    Technical Architecture:
    - Microservices architecture with Docker containers
    - AWS cloud infrastructure with auto-scaling
    - PostgreSQL primary database, Redis for caching
    - React frontend with TypeScript
    - CI/CD pipeline using GitHub Actions
    
    Business Requirements:
    - Customer onboarding must complete within 5 minutes
    - Support for 10,000+ concurrent users
    - GDPR and SOC2 compliance required
    - Multi-tenant architecture with data isolation
    
    Current Challenges:
    - Legacy authentication system causing bottlenecks
    - Manual verification processes slowing onboarding
    - Limited analytics on user behavior
    - Customer support tickets increasing 15% monthly
    
    Success Metrics:
    - Onboarding completion rate > 85%
    - Time to first value < 10 minutes
    - Customer satisfaction score > 4.5/5
    - Support ticket reduction by 30%
    """
    
    # Create RAG-enhanced team members
    team_members = [
        RAGTeamMember(
            "sarah-architect",
            "Technical Architect",
            "system design and scalability",
            company_context
        ),
        RAGTeamMember(
            "mike-product",
            "Product Manager", 
            "user experience and business requirements",
            company_context
        ),
        RAGTeamMember(
            "alex-security",
            "Security Engineer",
            "compliance and data protection",
            company_context
        ),
        RAGTeamMember(
            "jordan-devops",
            "DevOps Engineer",
            "infrastructure and deployment",
            company_context
        )
    ]
    
    # Setup LLM for each team member
    for member in team_members:
        member.setup_llm(config)
    
    # Create collaborative campfire
    team_campfire = Campfire("rag-enhanced-team")
    for member in team_members:
        team_campfire.add_camper(member)
    
    await team_campfire.start()
    
    # Collaborative decision making with RAG context
    decision_torch = Torch(
        claim="Design and implement an improved customer onboarding flow that addresses our current challenges",
        metadata={
            "project_priority": "high",
            "timeline": "8_weeks",
            "budget": "150k",
            "stakeholders": ["product", "engineering", "security", "operations"]
        }
    )
    
    await team_campfire.send_torch(decision_torch)
    await team_campfire.stop()
    
    print("RAG-enhanced team collaboration complete!")
    print("The HTML report demonstrates:")
    print("- How document context influenced each team member's analysis")
    print("- Role-specific perspectives and expertise contributions")
    print("- Evidence-based recommendations with source references")
    print("- Cross-functional collaboration insights")
    print("- Context utilization and information synthesis")

if __name__ == "__main__":
    asyncio.run(rag_team_collaboration_example())
```

## Basic LLM-Enabled Camper

A simple example showing the fundamental pattern for creating LLM-enabled Campers.

```python
import asyncio
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class BasicLLMCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        
        # Setup LLM configuration
        config = OpenRouterConfig()  # Uses environment variables
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Basic LLM interaction with custom prompt"""
        try:
            prompt = f"""
            You are a helpful AI assistant. Please provide a thoughtful and 
            informative response to the following question or request:
            
            {torch.claim}
            
            Provide a clear, concise, and helpful answer.
            """
            
            response = self.llm_completion_with_mcp(prompt)
            
            return {
                "claim": response,
                "confidence": 0.8,
                "metadata": {
                    "model_used": "llm",
                    "response_type": "general_assistance"
                }
            }
        except Exception as e:
            return {
                "claim": f"I apologize, but I'm unable to process your request right now: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True}
            }

# Usage example
async def basic_example():
    assistant = BasicLLMCamper("ai-assistant")
    campfire = Campfire("basic-demo")
    campfire.add_camper(assistant)
    
    await campfire.start()
    
    # Send a question
    torch = Torch(claim="What are the benefits of renewable energy?")
    result = await campfire.send_torch(torch)
    
    print(f"Response: {result.claim}")
    
    await campfire.stop()

if __name__ == "__main__":
    asyncio.run(basic_example())
```

## Expert Analysis System

Create domain experts that provide specialized analysis and recommendations.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class DomainExpert(Camper, LLMCamperMixin):
    def __init__(self, name: str, expertise: str, background: str):
        super().__init__(name)
        self.expertise = expertise
        self.background = background
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Expert analysis with domain-specific knowledge"""
        try:
            expert_prompt = f"""
            You are a world-class expert in {self.expertise}.
            
            Background: {self.background}
            
            Analysis Request: {torch.claim}
            
            Please provide an expert analysis including:
            1. Key insights from your domain expertise
            2. Potential risks or considerations
            3. Specific recommendations
            4. Confidence level in your analysis (0-100%)
            
            Be thorough but concise. Focus on actionable insights.
            """
            
            response = self.llm_completion_with_mcp(
                expert_prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract confidence from response (simplified)
            confidence = 0.85  # In practice, parse from response
            
            return {
                "claim": response,
                "confidence": confidence,
                "metadata": {
                    "expertise": self.expertise,
                    "analysis_type": "expert_domain",
                    "expert_name": self.name
                }
            }
        except Exception as e:
            return {
                "claim": f"Expert analysis unavailable: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "expertise": self.expertise}
            }

# Usage example
async def expert_analysis_example():
    # Create domain experts
    cybersecurity_expert = DomainExpert(
        "cyber-expert",
        "cybersecurity",
        "15+ years in enterprise security, CISSP certified, specializing in threat analysis and risk assessment"
    )
    
    finance_expert = DomainExpert(
        "finance-expert", 
        "financial analysis",
        "CFA charterholder with 12+ years in investment banking and risk management"
    )
    
    # Create expert panel
    expert_panel = Campfire("expert-analysis")
    expert_panel.add_camper(cybersecurity_expert)
    expert_panel.add_camper(finance_expert)
    
    await expert_panel.start()
    
    # Analyze a business decision
    torch = Torch(claim="Should our company implement a bring-your-own-device (BYOD) policy?")
    results = await expert_panel.send_torch(torch)
    
    print("Expert Analysis Results:")
    for result in results:
        print(f"\n{result.metadata.get('expertise', 'Unknown')}: {result.claim}")
    
    await expert_panel.stop()

if __name__ == "__main__":
    asyncio.run(expert_analysis_example())
```

## Content Generation Pipeline

A multi-stage content creation system with different specialized roles.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class ContentCreator(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, style_guide: str):
        super().__init__(name)
        self.role = role
        self.style_guide = style_guide
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Generate content based on role and style guidelines"""
        try:
            if self.role == "researcher":
                prompt = f"""
                You are a content researcher. Research and outline the following topic:
                
                Topic: {torch.claim}
                
                Provide:
                1. Key points to cover
                2. Target audience considerations
                3. Relevant statistics or facts
                4. Suggested structure
                
                Style Guide: {self.style_guide}
                """
            
            elif self.role == "writer":
                prompt = f"""
                You are a content writer. Create engaging content based on this outline:
                
                Content Brief: {torch.claim}
                
                Write compelling content that:
                1. Engages the target audience
                2. Follows the provided structure
                3. Incorporates key facts and statistics
                4. Maintains consistent tone and style
                
                Style Guide: {self.style_guide}
                """
            
            elif self.role == "editor":
                prompt = f"""
                You are a content editor. Review and improve this content:
                
                Content to Edit: {torch.claim}
                
                Focus on:
                1. Clarity and readability
                2. Grammar and style consistency
                3. Fact-checking and accuracy
                4. Overall flow and structure
                
                Style Guide: {self.style_guide}
                
                Provide the edited version with brief notes on changes made.
                """
            
            response = self.llm_completion_with_mcp(prompt)
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "role": self.role,
                    "content_stage": self.role,
                    "style_applied": True
                }
            }
        except Exception as e:
            return {
                "claim": f"Content generation failed for {self.role}: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "role": self.role}
            }

# Usage example
async def content_pipeline_example():
    style_guide = """
    - Professional but approachable tone
    - Use active voice
    - Include concrete examples
    - Target audience: business professionals
    - Length: 800-1200 words
    """
    
    # Create content team
    researcher = ContentCreator("researcher", "researcher", style_guide)
    writer = ContentCreator("writer", "writer", style_guide)
    editor = ContentCreator("editor", "editor", style_guide)
    
    # Sequential processing
    topic = "The Future of Remote Work in 2024"
    
    # Research phase
    research_campfire = Campfire("research")
    research_campfire.add_camper(researcher)
    await research_campfire.start()
    
    research_torch = Torch(claim=topic)
    research_result = await research_campfire.send_torch(research_torch)
    
    print("Research Phase Complete:")
    print(research_result.claim[:200] + "...")
    
    # Writing phase
    writing_campfire = Campfire("writing")
    writing_campfire.add_camper(writer)
    await writing_campfire.start()
    
    writing_torch = Torch(claim=research_result.claim)
    writing_result = await writing_campfire.send_torch(writing_torch)
    
    print("\nWriting Phase Complete:")
    print(writing_result.claim[:200] + "...")
    
    # Editing phase
    editing_campfire = Campfire("editing")
    editing_campfire.add_camper(editor)
    await editing_campfire.start()
    
    editing_torch = Torch(claim=writing_result.claim)
    final_result = await editing_campfire.send_torch(editing_torch)
    
    print("\nFinal Content:")
    print(final_result.claim)
    
    # Cleanup
    await research_campfire.stop()
    await writing_campfire.stop()
    await editing_campfire.stop()

if __name__ == "__main__":
    asyncio.run(content_pipeline_example())
```

## Multi-Agent Collaboration

Demonstrate how multiple LLM-enabled agents can collaborate on complex tasks.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class CollaborativeAgent(Camper, LLMCamperMixin):
    def __init__(self, name: str, role: str, collaboration_context: str):
        super().__init__(name)
        self.role = role
        self.collaboration_context = collaboration_context
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Collaborative response considering team context"""
        try:
            collaborative_prompt = f"""
            You are part of a collaborative team working on: {self.collaboration_context}
            
            Your role: {self.role}
            
            Current task/question: {torch.claim}
            
            Consider:
            1. How your expertise contributes to the team goal
            2. What other team members might need from you
            3. Dependencies or coordination required
            4. Your specific deliverables for this task
            
            Provide your contribution while being mindful of the collaborative nature.
            """
            
            response = self.llm_completion_with_mcp(collaborative_prompt)
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "role": self.role,
                    "collaboration_type": "team_contribution",
                    "context": self.collaboration_context
                }
            }
        except Exception as e:
            return {
                "claim": f"Unable to contribute as {self.role}: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "role": self.role}
            }

# Usage example
async def collaboration_example():
    project_context = "Developing a new mobile app for food delivery"
    
    # Create collaborative team
    product_manager = CollaborativeAgent(
        "product-manager",
        "Product Manager - defines requirements and user stories",
        project_context
    )
    
    ux_designer = CollaborativeAgent(
        "ux-designer", 
        "UX Designer - creates user experience and interface designs",
        project_context
    )
    
    tech_lead = CollaborativeAgent(
        "tech-lead",
        "Technical Lead - defines architecture and technical approach",
        project_context
    )
    
    qa_engineer = CollaborativeAgent(
        "qa-engineer",
        "QA Engineer - defines testing strategy and quality assurance",
        project_context
    )
    
    # Create collaborative campfire
    team_campfire = Campfire("app-development-team")
    team_campfire.add_camper(product_manager)
    team_campfire.add_camper(ux_designer)
    team_campfire.add_camper(tech_lead)
    team_campfire.add_camper(qa_engineer)
    
    await team_campfire.start()
    
    # Collaborative planning session
    planning_questions = [
        "What are the core features for our MVP?",
        "How should we handle user authentication and security?",
        "What's our approach for handling real-time order tracking?",
        "How do we ensure the app performs well under high load?"
    ]
    
    for question in planning_questions:
        print(f"\n{'='*60}")
        print(f"TEAM DISCUSSION: {question}")
        print('='*60)
        
        torch = Torch(claim=question)
        results = await team_campfire.send_torch(torch)
        
        for result in results:
            role = result.metadata.get('role', 'Unknown')
            print(f"\n{role.upper()}:")
            print(result.claim)
    
    await team_campfire.stop()

if __name__ == "__main__":
    asyncio.run(collaboration_example())
```

## Crisis Detection and Response

An intelligent system for detecting and responding to crisis situations.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire
import re

class CrisisDetectionCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Analyze content for crisis indicators"""
        try:
            crisis_prompt = f"""
            You are a crisis detection specialist. Analyze the following content for signs of:
            - Mental health crisis (depression, anxiety, suicidal ideation)
            - Safety concerns (violence, abuse, threats)
            - Emergency situations (medical, safety, urgent help needed)
            
            Content to analyze: {torch.claim}
            
            Provide:
            1. Crisis level (NONE, LOW, MEDIUM, HIGH, CRITICAL)
            2. Type of crisis detected (if any)
            3. Specific indicators found
            4. Recommended response level
            5. Confidence in assessment (0-100%)
            
            Be thorough but sensitive. Err on the side of caution for safety.
            """
            
            response = self.llm_completion_with_mcp(crisis_prompt)
            
            # Extract crisis level (simplified parsing)
            crisis_level = "MEDIUM"  # In practice, parse from response
            if "CRITICAL" in response.upper():
                crisis_level = "CRITICAL"
            elif "HIGH" in response.upper():
                crisis_level = "HIGH"
            elif "LOW" in response.upper():
                crisis_level = "LOW"
            elif "NONE" in response.upper():
                crisis_level = "NONE"
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "crisis_level": crisis_level,
                    "analysis_type": "crisis_detection",
                    "requires_escalation": crisis_level in ["HIGH", "CRITICAL"]
                }
            }
        except Exception as e:
            return {
                "claim": f"Crisis analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "crisis_level": "UNKNOWN"}
            }

class CrisisResponseCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str):
        super().__init__(name)
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Generate appropriate crisis response"""
        try:
            # Extract crisis level from metadata if available
            crisis_level = torch.metadata.get("crisis_level", "UNKNOWN")
            
            response_prompt = f"""
            You are a crisis response specialist. Based on the crisis analysis below,
            provide an appropriate, compassionate, and helpful response.
            
            Crisis Analysis: {torch.claim}
            Crisis Level: {crisis_level}
            
            Guidelines:
            - Be empathetic and non-judgmental
            - Provide immediate safety resources if needed
            - Offer practical next steps
            - Include relevant hotlines or emergency contacts
            - Maintain professional boundaries
            
            Provide a response that prioritizes safety and support.
            """
            
            response = self.llm_completion_with_mcp(response_prompt)
            
            return {
                "claim": response,
                "confidence": 0.95,
                "metadata": {
                    "response_type": "crisis_support",
                    "crisis_level": crisis_level,
                    "includes_resources": True
                }
            }
        except Exception as e:
            return {
                "claim": "I'm concerned about your wellbeing. Please reach out to a mental health professional or crisis hotline for immediate support.",
                "confidence": 0.8,
                "metadata": {"error": True, "fallback_response": True}
            }

# Usage example
async def crisis_response_example():
    # Create crisis response system
    detector = CrisisDetectionCamper("crisis-detector")
    responder = CrisisResponseCamper("crisis-responder")
    
    # Test scenarios
    test_messages = [
        "I'm feeling really overwhelmed with work lately",
        "I don't see the point in anything anymore and I'm thinking about ending it all",
        "My partner has been threatening me and I'm scared",
        "I'm having chest pains and trouble breathing"
    ]
    
    for message in test_messages:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {message}")
        print('='*60)
        
        # Detection phase
        detection_campfire = Campfire("crisis-detection")
        detection_campfire.add_camper(detector)
        await detection_campfire.start()
        
        detection_torch = Torch(claim=message)
        detection_result = await detection_campfire.send_torch(detection_torch)
        
        print(f"CRISIS ANALYSIS:")
        print(detection_result.claim)
        
        # Response phase
        response_campfire = Campfire("crisis-response")
        response_campfire.add_camper(responder)
        await response_campfire.start()
        
        response_torch = Torch(
            claim=detection_result.claim,
            metadata=detection_result.metadata
        )
        response_result = await response_campfire.send_torch(response_torch)
        
        print(f"\nCRISIS RESPONSE:")
        print(response_result.claim)
        
        await detection_campfire.stop()
        await response_campfire.stop()

if __name__ == "__main__":
    asyncio.run(crisis_response_example())
```

## Code Review Assistant

An intelligent code review system that provides detailed feedback and suggestions.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class CodeReviewCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, language: str, review_focus: str):
        super().__init__(name)
        self.language = language
        self.review_focus = review_focus
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Perform detailed code review"""
        try:
            review_prompt = f"""
            You are an expert {self.language} code reviewer with focus on {self.review_focus}.
            
            Please review the following code:
            
            ```{self.language}
            {torch.claim}
            ```
            
            Provide a comprehensive review including:
            
            1. **Code Quality Assessment**
               - Readability and maintainability
               - Code structure and organization
               - Naming conventions
            
            2. **{self.review_focus} Analysis**
               - Specific issues related to {self.review_focus}
               - Best practices compliance
               - Potential improvements
            
            3. **Bugs and Issues**
               - Logic errors or potential bugs
               - Edge cases not handled
               - Error handling gaps
            
            4. **Performance Considerations**
               - Efficiency concerns
               - Resource usage
               - Scalability issues
            
            5. **Recommendations**
               - Specific code improvements
               - Refactoring suggestions
               - Additional testing needs
            
            Rate the code quality from 1-10 and provide actionable feedback.
            """
            
            response = self.llm_completion_with_mcp(
                review_prompt,
                temperature=0.3,  # Lower temperature for more consistent reviews
                max_tokens=1500
            )
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "review_type": "code_review",
                    "language": self.language,
                    "focus": self.review_focus,
                    "reviewer": self.name
                }
            }
        except Exception as e:
            return {
                "claim": f"Code review failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "language": self.language}
            }

# Usage example
async def code_review_example():
    # Create specialized reviewers
    security_reviewer = CodeReviewCamper(
        "security-reviewer",
        "python",
        "security vulnerabilities and best practices"
    )
    
    performance_reviewer = CodeReviewCamper(
        "performance-reviewer",
        "python", 
        "performance optimization and efficiency"
    )
    
    maintainability_reviewer = CodeReviewCamper(
        "maintainability-reviewer",
        "python",
        "code maintainability and clean code principles"
    )
    
    # Sample code to review
    code_to_review = '''
def process_user_data(user_input):
    # Process user data
    data = eval(user_input)  # Potential security issue
    
    results = []
    for i in range(len(data)):  # Inefficient iteration
        if data[i] != None:  # Should use 'is not None'
            result = expensive_operation(data[i])
            results.append(result)
    
    return results

def expensive_operation(item):
    # Simulate expensive operation
    total = 0
    for i in range(1000000):
        total += i * item
    return total
'''
    
    # Create review panel
    review_panel = Campfire("code-review-panel")
    review_panel.add_camper(security_reviewer)
    review_panel.add_camper(performance_reviewer)
    review_panel.add_camper(maintainability_reviewer)
    
    await review_panel.start()
    
    print("CODE REVIEW SESSION")
    print("="*60)
    print("Code under review:")
    print(code_to_review)
    print("\n" + "="*60)
    
    # Submit code for review
    review_torch = Torch(claim=code_to_review)
    review_results = await review_panel.send_torch(review_torch)
    
    # Display reviews
    for result in review_results:
        reviewer_focus = result.metadata.get('focus', 'general')
        print(f"\n{reviewer_focus.upper()} REVIEW:")
        print("-" * 40)
        print(result.claim)
    
    await review_panel.stop()

if __name__ == "__main__":
    asyncio.run(code_review_example())
```

## Research and Summarization

An intelligent research assistant that can gather, analyze, and summarize information.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class ResearchCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, research_specialty: str):
        super().__init__(name)
        self.research_specialty = research_specialty
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Conduct specialized research and analysis"""
        try:
            research_prompt = f"""
            You are a research specialist in {self.research_specialty}.
            
            Research Topic: {torch.claim}
            
            Please provide a comprehensive research summary including:
            
            1. **Key Findings**
               - Most important discoveries or insights
               - Current state of knowledge
               - Recent developments
            
            2. **Data and Evidence**
               - Relevant statistics and metrics
               - Supporting evidence
               - Data sources and reliability
            
            3. **Different Perspectives**
               - Various viewpoints on the topic
               - Controversies or debates
               - Consensus areas
            
            4. **Implications**
               - Practical applications
               - Future implications
               - Impact on stakeholders
            
            5. **Research Gaps**
               - Areas needing more study
               - Unanswered questions
               - Future research directions
            
            Focus on {self.research_specialty} aspects while maintaining objectivity.
            """
            
            response = self.llm_completion_with_mcp(
                research_prompt,
                temperature=0.4,
                max_tokens=1200
            )
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "research_type": self.research_specialty,
                    "analysis_depth": "comprehensive",
                    "researcher": self.name
                }
            }
        except Exception as e:
            return {
                "claim": f"Research analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "specialty": self.research_specialty}
            }

class SummarizationCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, summary_style: str):
        super().__init__(name)
        self.summary_style = summary_style
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Create targeted summaries based on style"""
        try:
            summary_prompt = f"""
            You are a summarization expert specializing in {self.summary_style} summaries.
            
            Content to summarize: {torch.claim}
            
            Create a {self.summary_style} summary that:
            
            {"- Provides executive-level insights for decision makers" if self.summary_style == "executive" else ""}
            {"- Focuses on technical details and implementation aspects" if self.summary_style == "technical" else ""}
            {"- Emphasizes practical applications and actionable insights" if self.summary_style == "practical" else ""}
            {"- Highlights key findings in an accessible format" if self.summary_style == "general" else ""}
            
            Structure:
            1. Main takeaways (3-5 bullet points)
            2. Key details (relevant to {self.summary_style} audience)
            3. Recommendations or next steps
            
            Keep it concise but comprehensive for the target audience.
            """
            
            response = self.llm_completion_with_mcp(summary_prompt)
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "summary_style": self.summary_style,
                    "summary_type": "structured",
                    "target_audience": self.summary_style
                }
            }
        except Exception as e:
            return {
                "claim": f"Summarization failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "style": self.summary_style}
            }

# Usage example
async def research_summarization_example():
    # Create research team
    market_researcher = ResearchCamper(
        "market-researcher",
        "market trends and consumer behavior"
    )
    
    tech_researcher = ResearchCamper(
        "tech-researcher", 
        "technology trends and innovation"
    )
    
    # Create summarization team
    executive_summarizer = SummarizationCamper(
        "executive-summarizer",
        "executive"
    )
    
    technical_summarizer = SummarizationCamper(
        "technical-summarizer",
        "technical"
    )
    
    # Research phase
    research_topic = "The impact of artificial intelligence on small business operations"
    
    print(f"RESEARCH TOPIC: {research_topic}")
    print("="*60)
    
    # Conduct research
    research_campfire = Campfire("research-team")
    research_campfire.add_camper(market_researcher)
    research_campfire.add_camper(tech_researcher)
    
    await research_campfire.start()
    
    research_torch = Torch(claim=research_topic)
    research_results = await research_campfire.send_torch(research_torch)
    
    # Combine research findings
    combined_research = "\n\n".join([result.claim for result in research_results])
    
    print("RESEARCH FINDINGS:")
    for result in research_results:
        specialty = result.metadata.get('research_type', 'general')
        print(f"\n{specialty.upper()} RESEARCH:")
        print("-" * 30)
        print(result.claim[:300] + "...")
    
    # Summarization phase
    print("\n" + "="*60)
    print("SUMMARY GENERATION")
    print("="*60)
    
    summary_campfire = Campfire("summary-team")
    summary_campfire.add_camper(executive_summarizer)
    summary_campfire.add_camper(technical_summarizer)
    
    await summary_campfire.start()
    
    summary_torch = Torch(claim=combined_research)
    summary_results = await summary_campfire.send_torch(summary_torch)
    
    for result in summary_results:
        style = result.metadata.get('summary_style', 'general')
        print(f"\n{style.upper()} SUMMARY:")
        print("-" * 30)
        print(result.claim)
    
    await research_campfire.stop()
    await summary_campfire.stop()

if __name__ == "__main__":
    asyncio.run(research_summarization_example())
```

## Customer Support Bot

An intelligent customer support system with escalation capabilities.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class CustomerSupportCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, support_level: str, knowledge_base: dict):
        super().__init__(name)
        self.support_level = support_level
        self.knowledge_base = knowledge_base
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Provide customer support based on level and knowledge base"""
        try:
            # Extract relevant knowledge
            relevant_info = self._find_relevant_knowledge(torch.claim)
            
            if self.support_level == "tier1":
                support_prompt = f"""
                You are a Tier 1 customer support representative. You handle common questions
                and basic troubleshooting. Be friendly, helpful, and professional.
                
                Customer Question: {torch.claim}
                
                Relevant Knowledge Base Info:
                {relevant_info}
                
                Provide:
                1. A helpful response to the customer
                2. Step-by-step instructions if applicable
                3. Indicate if escalation to Tier 2 is needed
                
                If you cannot resolve the issue, recommend escalation.
                """
            
            elif self.support_level == "tier2":
                support_prompt = f"""
                You are a Tier 2 technical support specialist. You handle complex technical
                issues and provide advanced troubleshooting.
                
                Escalated Issue: {torch.claim}
                
                Technical Knowledge Base:
                {relevant_info}
                
                Provide:
                1. Advanced technical analysis
                2. Detailed troubleshooting steps
                3. Root cause analysis if possible
                4. Indicate if escalation to engineering is needed
                
                Be thorough and technical while remaining clear.
                """
            
            elif self.support_level == "specialist":
                support_prompt = f"""
                You are a specialist support engineer. You handle the most complex issues
                and provide expert-level technical guidance.
                
                Complex Issue: {torch.claim}
                
                Expert Knowledge Base:
                {relevant_info}
                
                Provide:
                1. Expert-level analysis and solution
                2. System-level recommendations
                3. Preventive measures
                4. Documentation for knowledge base updates
                
                Focus on comprehensive resolution and system improvement.
                """
            
            response = self.llm_completion_with_mcp(support_prompt)
            
            # Determine if escalation is needed
            needs_escalation = "escalat" in response.lower() and self.support_level != "specialist"
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "support_level": self.support_level,
                    "needs_escalation": needs_escalation,
                    "knowledge_used": bool(relevant_info),
                    "support_type": "customer_service"
                }
            }
        except Exception as e:
            return {
                "claim": f"I apologize, but I'm experiencing technical difficulties. Please try again or contact our technical team directly.",
                "confidence": 0.3,
                "metadata": {"error": True, "support_level": self.support_level}
            }
    
    def _find_relevant_knowledge(self, query: str) -> str:
        """Simple knowledge base search (in practice, use vector search)"""
        relevant = []
        query_lower = query.lower()
        
        for topic, info in self.knowledge_base.items():
            if any(keyword in query_lower for keyword in topic.split()):
                relevant.append(f"{topic}: {info}")
        
        return "\n".join(relevant[:3])  # Top 3 relevant entries

# Usage example
async def customer_support_example():
    # Knowledge base
    knowledge_base = {
        "login password reset": "Users can reset passwords via the 'Forgot Password' link on the login page. Reset emails are sent within 5 minutes.",
        "payment billing": "Billing issues can be resolved by checking the billing section in account settings. Contact billing@company.com for payment disputes.",
        "api rate limits": "API rate limits are 1000 requests per hour for free tier, 10000 for premium. Rate limit headers are included in responses.",
        "database connection": "Database connection issues often relate to firewall settings or connection string configuration. Check logs for specific error codes.",
        "ssl certificate": "SSL certificate errors require certificate renewal or configuration updates. Check certificate expiration dates first.",
        "server performance": "Performance issues may be related to memory usage, CPU load, or database query optimization. Monitor system metrics."
    }
    
    # Create support team
    tier1_agent = CustomerSupportCamper("tier1-agent", "tier1", knowledge_base)
    tier2_agent = CustomerSupportCamper("tier2-agent", "tier2", knowledge_base)
    specialist_agent = CustomerSupportCamper("specialist-agent", "specialist", knowledge_base)
    
    # Customer support scenarios
    support_cases = [
        "I can't log into my account, I forgot my password",
        "My API calls are being rejected with a 429 error",
        "The application is running very slowly and timing out",
        "I'm getting SSL certificate errors when connecting to your service"
    ]
    
    for case in support_cases:
        print(f"\n{'='*60}")
        print(f"CUSTOMER ISSUE: {case}")
        print('='*60)
        
        # Start with Tier 1
        tier1_campfire = Campfire("tier1-support")
        tier1_campfire.add_camper(tier1_agent)
        await tier1_campfire.start()
        
        support_torch = Torch(claim=case)
        tier1_result = await tier1_campfire.send_torch(support_torch)
        
        print("TIER 1 RESPONSE:")
        print(tier1_result.claim)
        
        # Check if escalation is needed
        if tier1_result.metadata.get("needs_escalation", False):
            print("\n--- ESCALATING TO TIER 2 ---")
            
            tier2_campfire = Campfire("tier2-support")
            tier2_campfire.add_camper(tier2_agent)
            await tier2_campfire.start()
            
            escalation_torch = Torch(
                claim=f"Escalated from Tier 1: {case}\n\nTier 1 Response: {tier1_result.claim}"
            )
            tier2_result = await tier2_campfire.send_torch(escalation_torch)
            
            print("TIER 2 RESPONSE:")
            print(tier2_result.claim)
            
            # Check if further escalation is needed
            if tier2_result.metadata.get("needs_escalation", False):
                print("\n--- ESCALATING TO SPECIALIST ---")
                
                specialist_campfire = Campfire("specialist-support")
                specialist_campfire.add_camper(specialist_agent)
                await specialist_campfire.start()
                
                specialist_torch = Torch(
                    claim=f"Escalated from Tier 2: {case}\n\nPrevious Analysis: {tier2_result.claim}"
                )
                specialist_result = await specialist_campfire.send_torch(specialist_torch)
                
                print("SPECIALIST RESPONSE:")
                print(specialist_result.claim)
                
                await specialist_campfire.stop()
            
            await tier2_campfire.stop()
        
        await tier1_campfire.stop()

if __name__ == "__main__":
    asyncio.run(customer_support_example())
```

## Data Analysis Agent

An intelligent data analysis system that can interpret and analyze data patterns.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire
import json

class DataAnalysisCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, analysis_type: str):
        super().__init__(name)
        self.analysis_type = analysis_type
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Perform data analysis based on type"""
        try:
            if self.analysis_type == "statistical":
                analysis_prompt = f"""
                You are a statistical data analyst. Analyze the following data:
                
                Data: {torch.claim}
                
                Provide statistical analysis including:
                1. Descriptive statistics (mean, median, mode, std dev)
                2. Data distribution characteristics
                3. Outliers and anomalies
                4. Correlation patterns
                5. Statistical significance of findings
                6. Confidence intervals where applicable
                
                Present findings in a clear, professional format.
                """
            
            elif self.analysis_type == "trend":
                analysis_prompt = f"""
                You are a trend analysis specialist. Analyze the following data for trends:
                
                Data: {torch.claim}
                
                Identify and analyze:
                1. Overall trends (upward, downward, cyclical)
                2. Seasonal patterns
                3. Growth rates and acceleration
                4. Trend reversals or inflection points
                5. Forecasting implications
                6. Factors driving trends
                
                Provide actionable insights for decision making.
                """
            
            elif self.analysis_type == "comparative":
                analysis_prompt = f"""
                You are a comparative data analyst. Analyze the following data for comparisons:
                
                Data: {torch.claim}
                
                Perform comparative analysis including:
                1. Performance comparisons between groups/periods
                2. Relative strengths and weaknesses
                3. Benchmark analysis
                4. Gap analysis
                5. Competitive positioning
                6. Improvement opportunities
                
                Highlight key differences and their implications.
                """
            
            response = self.llm_completion_with_mcp(
                analysis_prompt,
                temperature=0.2,  # Lower temperature for more consistent analysis
                max_tokens=1200
            )
            
            return {
                "claim": response,
                "confidence": 0.9,
                "metadata": {
                    "analysis_type": self.analysis_type,
                    "data_analysis": True,
                    "analyst": self.name
                }
            }
        except Exception as e:
            return {
                "claim": f"Data analysis failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "analysis_type": self.analysis_type}
            }

# Usage example
async def data_analysis_example():
    # Sample data for analysis
    sales_data = {
        "monthly_sales": [120000, 135000, 142000, 138000, 155000, 168000, 
                         172000, 165000, 178000, 185000, 192000, 198000],
        "regions": {
            "North": [45000, 48000, 52000, 49000, 58000, 62000],
            "South": [35000, 42000, 45000, 44000, 48000, 52000],
            "East": [25000, 28000, 30000, 29000, 32000, 35000],
            "West": [15000, 17000, 15000, 16000, 17000, 19000]
        },
        "products": {
            "Product A": {"sales": 850000, "units": 1200, "growth": 15.2},
            "Product B": {"sales": 620000, "units": 890, "growth": 8.7},
            "Product C": {"sales": 420000, "units": 650, "growth": -2.1},
            "Product D": {"sales": 380000, "units": 520, "growth": 22.5}
        }
    }
    
    # Create analysis team
    statistical_analyst = DataAnalysisCamper("statistical-analyst", "statistical")
    trend_analyst = DataAnalysisCamper("trend-analyst", "trend")
    comparative_analyst = DataAnalysisCamper("comparative-analyst", "comparative")
    
    # Create analysis campfire
    analysis_campfire = Campfire("data-analysis-team")
    analysis_campfire.add_camper(statistical_analyst)
    analysis_campfire.add_camper(trend_analyst)
    analysis_campfire.add_camper(comparative_analyst)
    
    await analysis_campfire.start()
    
    print("DATA ANALYSIS SESSION")
    print("="*60)
    print("Dataset:")
    print(json.dumps(sales_data, indent=2))
    print("\n" + "="*60)
    
    # Perform analysis
    data_torch = Torch(claim=json.dumps(sales_data, indent=2))
    analysis_results = await analysis_campfire.send_torch(data_torch)
    
    # Display analysis results
    for result in analysis_results:
        analysis_type = result.metadata.get('analysis_type', 'general')
        print(f"\n{analysis_type.upper()} ANALYSIS:")
        print("-" * 40)
        print(result.claim)
    
    await analysis_campfire.stop()

if __name__ == "__main__":
    asyncio.run(data_analysis_example())
```

## Creative Writing Assistant

An intelligent creative writing system with different writing styles and genres.

```python
from campfires import Camper, LLMCamperMixin, OpenRouterConfig, Torch, Campfire

class CreativeWriterCamper(Camper, LLMCamperMixin):
    def __init__(self, name: str, writing_style: str, genre: str):
        super().__init__(name)
        self.writing_style = writing_style
        self.genre = genre
        
        config = OpenRouterConfig()
        self.setup_llm(config)
    
    def override_prompt(self, torch: Torch) -> dict:
        """Generate creative content based on style and genre"""
        try:
            creative_prompt = f"""
            You are a creative writer specializing in {self.genre} with a {self.writing_style} style.
            
            Writing Request: {torch.claim}
            
            Style Guidelines for {self.writing_style}:
            {self._get_style_guidelines()}
            
            Genre Conventions for {self.genre}:
            {self._get_genre_conventions()}
            
            Create engaging, original content that:
            1. Captures the essence of the {self.writing_style} style
            2. Follows {self.genre} conventions
            3. Engages the reader from the beginning
            4. Maintains consistent tone and voice
            5. Includes vivid descriptions and compelling characters
            
            Aim for approximately 300-500 words unless otherwise specified.
            """
            
            response = self.llm_completion_with_mcp(
                creative_prompt,
                temperature=0.8,  # Higher temperature for creativity
                max_tokens=800
            )
            
            return {
                "claim": response,
                "confidence": 0.85,
                "metadata": {
                    "writing_style": self.writing_style,
                    "genre": self.genre,
                    "content_type": "creative_writing",
                    "writer": self.name
                }
            }
        except Exception as e:
            return {
                "claim": f"Creative writing failed: {str(e)}",
                "confidence": 0.1,
                "metadata": {"error": True, "style": self.writing_style}
            }
    
    def _get_style_guidelines(self) -> str:
        """Get style-specific guidelines"""
        styles = {
            "literary": "Focus on character development, symbolism, and deeper themes. Use rich, descriptive language.",
            "minimalist": "Use sparse, precise language. Focus on subtext and what's not said. Economy of words.",
            "experimental": "Break conventional narrative structures. Play with form, perspective, and language.",
            "commercial": "Prioritize plot and pacing. Clear, accessible language. Strong hooks and cliffhangers.",
            "poetic": "Emphasize rhythm, imagery, and metaphor. Language should be musical and evocative."
        }
        return styles.get(self.writing_style, "Write in a clear, engaging style appropriate for the genre.")
    
    def _get_genre_conventions(self) -> str:
        """Get genre-specific conventions"""
        genres = {
            "science_fiction": "Explore futuristic concepts, technology, and their impact on society. Build believable worlds.",
            "fantasy": "Create magical worlds with consistent rules. Focus on world-building and mythic elements.",
            "mystery": "Build suspense through clues and red herrings. Fair play with the reader.",
            "romance": "Focus on relationship development and emotional connection. Satisfying romantic arc.",
            "horror": "Build atmosphere and tension. Use fear and suspense effectively.",
            "literary_fiction": "Explore human condition and complex themes. Character-driven narratives.",
            "thriller": "Maintain high tension and fast pacing. Constant sense of danger or urgency."
        }
        return genres.get(self.genre, "Follow standard narrative conventions with strong character and plot development.")

# Usage example
async def creative_writing_example():
    # Create diverse writing team
    literary_writer = CreativeWriterCamper(
        "literary-writer",
        "literary",
        "literary_fiction"
    )
    
    commercial_writer = CreativeWriterCamper(
        "commercial-writer",
        "commercial", 
        "thriller"
    )
    
    experimental_writer = CreativeWriterCamper(
        "experimental-writer",
        "experimental",
        "science_fiction"
    )
    
    poetic_writer = CreativeWriterCamper(
        "poetic-writer",
        "poetic",
        "fantasy"
    )
    
    # Writing prompts
    writing_prompts = [
        "Write about a character who discovers they can hear other people's thoughts, but only their regrets",
        "A detective realizes the serial killer they're hunting is their future self",
        "In a world where memories can be extracted and sold, a memory thief discovers their own stolen past",
        "A librarian finds that the books in their library are changing their stories on their own"
    ]
    
    for i, prompt in enumerate(writing_prompts):
        print(f"\n{'='*80}")
        print(f"WRITING PROMPT {i+1}: {prompt}")
        print('='*80)
        
        # Select writer based on prompt
        if i == 0:
            writer = literary_writer
        elif i == 1:
            writer = commercial_writer
        elif i == 2:
            writer = experimental_writer
        else:
            writer = poetic_writer
        
        # Create writing session
        writing_campfire = Campfire(f"writing-session-{i+1}")
        writing_campfire.add_camper(writer)
        
        await writing_campfire.start()
        
        writing_torch = Torch(claim=prompt)
        writing_result = await writing_campfire.send_torch(writing_torch)
        
        style = writing_result.metadata.get('writing_style', 'unknown')
        genre = writing_result.metadata.get('genre', 'unknown')
        
        print(f"\n{style.upper()} {genre.upper()} STORY:")
        print("-" * 50)
        print(writing_result.claim)
        
        await writing_campfire.stop()
    
    # Collaborative writing session
    print(f"\n{'='*80}")
    print("COLLABORATIVE WRITING SESSION")
    print('='*80)
    
    collaborative_campfire = Campfire("collaborative-writing")
    collaborative_campfire.add_camper(literary_writer)
    collaborative_campfire.add_camper(commercial_writer)
    collaborative_campfire.add_camper(experimental_writer)
    collaborative_campfire.add_camper(poetic_writer)
    
    await collaborative_campfire.start()
    
    collaborative_prompt = "Write the opening paragraph of a story about a time traveler who gets stuck in a time loop, but each writer should interpret this in their own style"
    
    collab_torch = Torch(claim=collaborative_prompt)
    collab_results = await collaborative_campfire.send_torch(collab_torch)
    
    print("COLLABORATIVE INTERPRETATIONS:")
    for result in collab_results:
        style = result.metadata.get('writing_style', 'unknown')
        genre = result.metadata.get('genre', 'unknown')
        print(f"\n{style.upper()} {genre.upper()} VERSION:")
        print("-" * 30)
        print(result.claim)
    
    await collaborative_campfire.stop()

if __name__ == "__main__":
    asyncio.run(creative_writing_example())
```

## Running the Examples

To run any of these examples:

1. **Set up environment variables** in a `.env` file:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_DEFAULT_MODEL=anthropic/claude-3-5-sonnet-20241022
   ```

2. **Install dependencies**:
   ```bash
   pip install campfires python-dotenv
   ```

3. **Run an example**:
   ```bash
   python example_name.py
   ```

## Key Patterns Demonstrated

1. **LLMCamperMixin Integration**: All examples show how to inherit from both `Camper` and `LLMCamperMixin`
2. **override_prompt Usage**: Each example demonstrates custom prompt engineering
3. **Error Handling**: Robust error handling patterns for LLM failures
4. **Metadata Usage**: Rich metadata for tracking and analysis
5. **Multi-Agent Collaboration**: Examples of agents working together
6. **Specialized Roles**: Domain-specific expertise and responsibilities
7. **Configuration Management**: Proper OpenRouter configuration
8. **Scalable Patterns**: Patterns that work for simple and complex use cases

These examples provide a comprehensive foundation for building sophisticated LLM-enabled applications using the Campfires framework.