"""
Multimodal prompt templates and engineering patterns for Campfires.
"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from jinja2 import Template


class PromptType(Enum):
    """Types of multimodal prompts."""
    VISION_ANALYSIS = "vision_analysis"
    IMAGE_COMPARISON = "image_comparison"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    CONTENT_DESCRIPTION = "content_description"
    MULTIMODAL_QA = "multimodal_qa"
    CREATIVE_GENERATION = "creative_generation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    ACCESSIBILITY = "accessibility"


@dataclass
class PromptTemplate:
    """A multimodal prompt template."""
    name: str
    type: PromptType
    template: str
    description: str
    required_content_types: List[str]
    optional_parameters: Dict[str, Any]
    examples: List[Dict[str, Any]]


class MultimodalPromptLibrary:
    """
    Library of multimodal prompt templates and engineering patterns.
    """
    
    def __init__(self):
        """Initialize the prompt library."""
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        
        # Vision Analysis Templates
        self.add_template(PromptTemplate(
            name="detailed_image_analysis",
            type=PromptType.VISION_ANALYSIS,
            template="""
Analyze this image in detail. Please provide:

1. **Visual Description**: Describe what you see in the image, including:
   - Main subjects and objects
   - Colors, lighting, and composition
   - Setting and environment
   - Any text visible in the image

2. **Technical Details**: 
   - Image quality and resolution assessment
   - Photographic techniques used (if applicable)
   - Any notable visual elements or patterns

3. **Context and Interpretation**:
   - What story or message does this image convey?
   - What might be the purpose or intent behind this image?
   - Any cultural, historical, or contextual significance

{% if focus_areas %}
**Special Focus**: Please pay particular attention to: {{ focus_areas | join(', ') }}
{% endif %}

{% if analysis_depth %}
**Analysis Depth**: {{ analysis_depth }}
{% endif %}
            """.strip(),
            description="Comprehensive analysis of images with structured output",
            required_content_types=["image"],
            optional_parameters={
                "focus_areas": "List of specific areas to focus on",
                "analysis_depth": "Level of detail (basic, detailed, expert)"
            },
            examples=[
                {
                    "input": "Photo of a sunset over mountains",
                    "parameters": {"focus_areas": ["lighting", "composition"], "analysis_depth": "detailed"},
                    "expected_output": "Detailed analysis focusing on lighting and composition"
                }
            ]
        ))
        
        self.add_template(PromptTemplate(
            name="image_comparison",
            type=PromptType.IMAGE_COMPARISON,
            template="""
Compare these images and provide a detailed analysis:

**Comparison Framework**:
1. **Visual Similarities**: What elements do these images share?
2. **Key Differences**: What distinguishes one image from another?
3. **Quality Assessment**: Compare technical quality, resolution, and clarity
4. **Style and Composition**: Compare artistic style, framing, and composition
5. **Content Analysis**: Compare subjects, themes, and messages

{% if comparison_criteria %}
**Specific Criteria**: Focus your comparison on: {{ comparison_criteria | join(', ') }}
{% endif %}

{% if output_format %}
**Output Format**: {{ output_format }}
{% endif %}

Please provide a structured comparison that highlights both similarities and differences.
            """.strip(),
            description="Compare multiple images with structured analysis",
            required_content_types=["image"],
            optional_parameters={
                "comparison_criteria": "Specific aspects to compare",
                "output_format": "Desired output format (table, list, narrative)"
            },
            examples=[
                {
                    "input": "Two photos of the same building taken at different times",
                    "parameters": {"comparison_criteria": ["lighting", "season", "architectural details"]},
                    "expected_output": "Structured comparison focusing on specified criteria"
                }
            ]
        ))
        
        # Audio Analysis Templates
        self.add_template(PromptTemplate(
            name="audio_content_analysis",
            type=PromptType.AUDIO_TRANSCRIPTION,
            template="""
Analyze this audio content and provide:

1. **Transcription**: Full text transcription of spoken content
2. **Audio Quality**: Assessment of recording quality, clarity, and any issues
3. **Speaker Analysis**: 
   - Number of speakers (if multiple)
   - Speaking style and tone
   - Accent or language characteristics
4. **Content Summary**: Key points and main themes discussed
5. **Technical Details**: 
   - Audio format and quality
   - Background noise or music
   - Any technical issues

{% if transcription_style %}
**Transcription Style**: {{ transcription_style }}
{% endif %}

{% if include_timestamps %}
**Include Timestamps**: Yes - provide timestamps for major sections
{% endif %}

{% if focus_on %}
**Special Focus**: Pay particular attention to: {{ focus_on }}
{% endif %}
            """.strip(),
            description="Comprehensive audio content analysis and transcription",
            required_content_types=["audio"],
            optional_parameters={
                "transcription_style": "Style of transcription (verbatim, clean, summary)",
                "include_timestamps": "Whether to include timestamps",
                "focus_on": "Specific aspects to focus on"
            },
            examples=[
                {
                    "input": "Recording of a business meeting",
                    "parameters": {"transcription_style": "clean", "include_timestamps": True},
                    "expected_output": "Clean transcription with timestamps and analysis"
                }
            ]
        ))
        
        # Multimodal QA Templates
        self.add_template(PromptTemplate(
            name="multimodal_qa",
            type=PromptType.MULTIMODAL_QA,
            template="""
Based on the provided content (images, audio, text), please answer the following question:

**Question**: {{ question }}

**Instructions**:
- Use information from ALL provided content types
- Cite specific elements from each content type in your answer
- If content types contradict each other, note the discrepancies
- Provide a comprehensive answer that synthesizes information across modalities

{% if answer_format %}
**Answer Format**: {{ answer_format }}
{% endif %}

{% if confidence_level %}
**Include Confidence**: Please indicate your confidence level in the answer
{% endif %}

{% if additional_context %}
**Additional Context**: {{ additional_context }}
{% endif %}
            """.strip(),
            description="Answer questions using multiple content types",
            required_content_types=["text"],  # At minimum, needs the question as text
            optional_parameters={
                "question": "The question to answer",
                "answer_format": "Desired format for the answer",
                "confidence_level": "Whether to include confidence assessment",
                "additional_context": "Any additional context for the question"
            },
            examples=[
                {
                    "input": "Image of a recipe + audio of cooking instructions",
                    "parameters": {"question": "What are the key steps to make this dish?"},
                    "expected_output": "Comprehensive answer using both visual and audio information"
                }
            ]
        ))
        
        # Creative Generation Templates
        self.add_template(PromptTemplate(
            name="creative_story_from_media",
            type=PromptType.CREATIVE_GENERATION,
            template="""
Create a {{ story_type }} based on the provided media content.

**Creative Brief**:
- Use the provided content as inspiration
- Incorporate visual, audio, or textual elements from the source material
- Create an original narrative that captures the essence of the content

{% if genre %}
**Genre**: {{ genre }}
{% endif %}

{% if target_audience %}
**Target Audience**: {{ target_audience }}
{% endif %}

{% if length %}
**Length**: {{ length }}
{% endif %}

{% if style %}
**Writing Style**: {{ style }}
{% endif %}

**Requirements**:
1. Reference specific elements from the source content
2. Maintain consistency with the mood and tone of the original
3. Create engaging, original content
4. Include vivid descriptions that bring the story to life

{% if additional_requirements %}
**Additional Requirements**: {{ additional_requirements }}
{% endif %}
            """.strip(),
            description="Generate creative content inspired by multimodal input",
            required_content_types=["image", "audio", "text"],
            optional_parameters={
                "story_type": "Type of story (short story, poem, script, etc.)",
                "genre": "Genre preference",
                "target_audience": "Intended audience",
                "length": "Desired length",
                "style": "Writing style preference",
                "additional_requirements": "Any additional creative requirements"
            },
            examples=[
                {
                    "input": "Photo of an old lighthouse + audio of ocean waves",
                    "parameters": {"story_type": "short story", "genre": "mystery", "length": "500 words"},
                    "expected_output": "Original mystery story inspired by the lighthouse and ocean sounds"
                }
            ]
        ))
        
        # Technical Analysis Templates
        self.add_template(PromptTemplate(
            name="technical_document_analysis",
            type=PromptType.TECHNICAL_ANALYSIS,
            template="""
Perform a technical analysis of the provided content:

**Analysis Scope**:
1. **Content Structure**: Analyze the organization and structure
2. **Technical Accuracy**: Assess technical correctness and completeness
3. **Quality Metrics**: Evaluate quality indicators
4. **Compliance**: Check against standards or requirements
5. **Recommendations**: Suggest improvements or modifications

{% if analysis_framework %}
**Framework**: Use {{ analysis_framework }} methodology
{% endif %}

{% if technical_domain %}
**Domain Focus**: {{ technical_domain }}
{% endif %}

{% if output_format %}
**Report Format**: {{ output_format }}
{% endif %}

**Deliverables**:
- Executive summary
- Detailed findings
- Risk assessment (if applicable)
- Actionable recommendations
- Quality score or rating

{% if specific_criteria %}
**Evaluation Criteria**: {{ specific_criteria | join(', ') }}
{% endif %}
            """.strip(),
            description="Technical analysis of documents, images, or audio content",
            required_content_types=["text", "image", "audio"],
            optional_parameters={
                "analysis_framework": "Technical analysis framework to use",
                "technical_domain": "Specific technical domain",
                "output_format": "Format for the analysis report",
                "specific_criteria": "Specific criteria to evaluate"
            },
            examples=[
                {
                    "input": "Technical diagram + specification document",
                    "parameters": {"technical_domain": "software architecture", "analysis_framework": "ISO 25010"},
                    "expected_output": "Comprehensive technical analysis report"
                }
            ]
        ))
        
        # Accessibility Templates
        self.add_template(PromptTemplate(
            name="accessibility_description",
            type=PromptType.ACCESSIBILITY,
            template="""
Create accessibility descriptions for the provided content:

**Accessibility Requirements**:
1. **Alt Text**: Concise, descriptive alternative text
2. **Long Description**: Detailed description for complex content
3. **Audio Description**: Describe visual elements for audio narration
4. **Transcript**: Full transcript for audio content
5. **Navigation Aids**: Structural information for screen readers

{% if accessibility_standard %}
**Standard**: Follow {{ accessibility_standard }} guidelines
{% endif %}

{% if target_disability %}
**Primary Focus**: Optimize for {{ target_disability }}
{% endif %}

{% if content_purpose %}
**Content Purpose**: {{ content_purpose }}
{% endif %}

**Output Requirements**:
- Clear, concise language
- Logical reading order
- Essential information prioritized
- Context-appropriate level of detail
- Inclusive language throughout

{% if additional_formats %}
**Additional Formats**: {{ additional_formats | join(', ') }}
{% endif %}
            """.strip(),
            description="Generate accessibility descriptions and aids",
            required_content_types=["image", "audio"],
            optional_parameters={
                "accessibility_standard": "Accessibility standard to follow (WCAG, Section 508, etc.)",
                "target_disability": "Primary disability to optimize for",
                "content_purpose": "Purpose of the content",
                "additional_formats": "Additional accessible formats needed"
            },
            examples=[
                {
                    "input": "Infographic about climate change",
                    "parameters": {"accessibility_standard": "WCAG 2.1 AA", "target_disability": "visual impairment"},
                    "expected_output": "Comprehensive accessibility descriptions following WCAG guidelines"
                }
            ]
        ))
    
    def add_template(self, template: PromptTemplate):
        """Add a template to the library."""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self, content_type: str = None, prompt_type: PromptType = None) -> List[PromptTemplate]:
        """List templates, optionally filtered by content type or prompt type."""
        templates = list(self.templates.values())
        
        if content_type:
            templates = [t for t in templates if content_type in t.required_content_types]
        
        if prompt_type:
            templates = [t for t in templates if t.type == prompt_type]
        
        return templates
    
    def list_template_names(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())
    
    def render_prompt(self, template_name: str, parameters: Dict[str, Any] = None) -> str:
        """
        Render a prompt template with parameters.
        
        Args:
            template_name: Name of the template
            parameters: Parameters to substitute in the template
            
        Returns:
            Rendered prompt string
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        jinja_template = Template(template.template)
        return jinja_template.render(**(parameters or {}))


class PromptEngineeringPatterns:
    """
    Collection of prompt engineering patterns for multimodal content.
    """
    
    @staticmethod
    def chain_of_thought_multimodal(content_types: List[str], task: str) -> str:
        """
        Generate a chain-of-thought prompt for multimodal analysis.
        
        Args:
            content_types: Types of content being analyzed
            task: The task to perform
            
        Returns:
            Chain-of-thought prompt
        """
        content_analysis = []
        
        for content_type in content_types:
            if content_type == "image":
                content_analysis.append("First, I'll analyze the visual content: What do I see in the image? What are the key visual elements?")
            elif content_type == "audio":
                content_analysis.append("Next, I'll analyze the audio content: What do I hear? What information is conveyed through sound?")
            elif content_type == "text":
                content_analysis.append("Then, I'll analyze the textual content: What information is provided in the text? What are the key points?")
        
        synthesis_step = "Finally, I'll synthesize information from all content types to complete the task."
        
        prompt = f"""
Let me approach this {task} step by step:

{chr(10).join(content_analysis)}

{synthesis_step}

Now, let me work through this systematically:
        """.strip()
        
        return prompt
    
    @staticmethod
    def few_shot_multimodal(examples: List[Dict[str, Any]], task: str) -> str:
        """
        Generate a few-shot learning prompt for multimodal tasks.
        
        Args:
            examples: List of example inputs and outputs
            task: The task to perform
            
        Returns:
            Few-shot prompt
        """
        prompt_parts = [f"Here are examples of {task}:"]
        
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {example.get('input', 'N/A')}")
            prompt_parts.append(f"Output: {example.get('output', 'N/A')}")
        
        prompt_parts.append(f"\nNow, please perform the same {task} for the provided content:")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def role_based_analysis(role: str, content_types: List[str], task: str) -> str:
        """
        Generate a role-based analysis prompt.
        
        Args:
            role: The role to assume (e.g., "art critic", "technical analyst")
            content_types: Types of content being analyzed
            task: The task to perform
            
        Returns:
            Role-based prompt
        """
        content_desc = ", ".join(content_types)
        
        prompt = f"""
As a {role}, please analyze the provided {content_desc} and {task}.

Approach this from the perspective of a {role}, considering:
- The standards and criteria relevant to your field
- The terminology and concepts specific to your expertise
- The quality indicators that matter in your domain
- The context and implications from your professional viewpoint

Please provide your analysis in a manner consistent with your role as a {role}.
        """.strip()
        
        return prompt
    
    @staticmethod
    def comparative_analysis(comparison_type: str, content_types: List[str]) -> str:
        """
        Generate a comparative analysis prompt.
        
        Args:
            comparison_type: Type of comparison (similarity, difference, evolution, etc.)
            content_types: Types of content being compared
            
        Returns:
            Comparative analysis prompt
        """
        content_desc = " and ".join(content_types)
        
        prompt = f"""
Perform a {comparison_type} analysis of the provided {content_desc}.

Structure your analysis as follows:

1. **Initial Assessment**: Briefly describe each piece of content
2. **Comparison Framework**: Establish the criteria for comparison
3. **Detailed Analysis**: Compare each aspect systematically
4. **Key Findings**: Highlight the most significant {comparison_type}s
5. **Conclusion**: Summarize your overall assessment

Focus on providing specific, evidence-based comparisons rather than general observations.
        """.strip()
        
        return prompt
    
    @staticmethod
    def progressive_refinement(initial_task: str, refinement_steps: List[str]) -> str:
        """
        Generate a progressive refinement prompt.
        
        Args:
            initial_task: The initial task to perform
            refinement_steps: List of refinement steps
            
        Returns:
            Progressive refinement prompt
        """
        prompt_parts = [f"Let's approach this {initial_task} through progressive refinement:"]
        
        prompt_parts.append(f"\nStep 1: Initial Analysis")
        prompt_parts.append(f"Perform a basic {initial_task} to establish the foundation.")
        
        for i, step in enumerate(refinement_steps, 2):
            prompt_parts.append(f"\nStep {i}: {step}")
            prompt_parts.append(f"Refine your analysis by focusing on {step.lower()}.")
        
        prompt_parts.append(f"\nFinal Step: Integration")
        prompt_parts.append("Integrate all refinements into a comprehensive final analysis.")
        
        return "\n".join(prompt_parts)


class MultimodalPromptBuilder:
    """
    Builder class for constructing complex multimodal prompts.
    """
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.components = []
        self.parameters = {}
    
    def add_instruction(self, instruction: str) -> 'MultimodalPromptBuilder':
        """Add an instruction component."""
        self.components.append(("instruction", instruction))
        return self
    
    def add_context(self, context: str) -> 'MultimodalPromptBuilder':
        """Add context information."""
        self.components.append(("context", context))
        return self
    
    def add_content_analysis(self, content_type: str, analysis_type: str) -> 'MultimodalPromptBuilder':
        """Add content analysis component."""
        analysis_text = f"Analyze the {content_type} content with focus on {analysis_type}."
        self.components.append(("analysis", analysis_text))
        return self
    
    def add_output_format(self, format_spec: str) -> 'MultimodalPromptBuilder':
        """Add output format specification."""
        self.components.append(("format", f"Output format: {format_spec}"))
        return self
    
    def add_constraints(self, constraints: List[str]) -> 'MultimodalPromptBuilder':
        """Add constraints or requirements."""
        constraints_text = "Constraints:\n" + "\n".join(f"- {c}" for c in constraints)
        self.components.append(("constraints", constraints_text))
        return self
    
    def add_examples(self, examples: List[Dict[str, Any]]) -> 'MultimodalPromptBuilder':
        """Add examples."""
        examples_text = "Examples:\n"
        for i, example in enumerate(examples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Input: {example.get('input', 'N/A')}\n"
            examples_text += f"Output: {example.get('output', 'N/A')}\n"
        self.components.append(("examples", examples_text))
        return self
    
    def set_parameter(self, key: str, value: Any) -> 'MultimodalPromptBuilder':
        """Set a parameter for template rendering."""
        self.parameters[key] = value
        return self
    
    def build(self) -> str:
        """Build the final prompt."""
        prompt_parts = []
        
        # Organize components by type
        component_order = ["context", "instruction", "examples", "analysis", "constraints", "format"]
        component_dict = {comp_type: [] for comp_type in component_order}
        
        for comp_type, content in self.components:
            if comp_type in component_dict:
                component_dict[comp_type].append(content)
        
        # Build prompt in logical order
        for comp_type in component_order:
            if component_dict[comp_type]:
                prompt_parts.extend(component_dict[comp_type])
                prompt_parts.append("")  # Add spacing
        
        # Remove trailing empty line
        if prompt_parts and prompt_parts[-1] == "":
            prompt_parts.pop()
        
        prompt = "\n".join(prompt_parts)
        
        # Apply parameter substitution if needed
        if self.parameters:
            template = Template(prompt)
            prompt = template.render(**self.parameters)
        
        return prompt
    
    def reset(self) -> 'MultimodalPromptBuilder':
        """Reset the builder for reuse."""
        self.components = []
        self.parameters = {}
        return self


# Global instance for easy access
prompt_library = MultimodalPromptLibrary()
prompt_patterns = PromptEngineeringPatterns()


def get_prompt_for_content_types(content_types: List[str], task_type: str = "analysis") -> str:
    """
    Get an appropriate prompt template for given content types and task.
    
    Args:
        content_types: List of content types (image, audio, text, etc.)
        task_type: Type of task to perform
        
    Returns:
        Appropriate prompt template
    """
    # Simple heuristic to select appropriate template
    if "image" in content_types and len(content_types) == 1:
        if task_type == "analysis":
            return prompt_library.render_prompt("detailed_image_analysis")
        elif task_type == "accessibility":
            return prompt_library.render_prompt("accessibility_description")
    
    elif "audio" in content_types and len(content_types) == 1:
        return prompt_library.render_prompt("audio_content_analysis")
    
    elif len(content_types) > 1:
        if task_type == "qa":
            return prompt_library.render_prompt("multimodal_qa")
        elif task_type == "creative":
            return prompt_library.render_prompt("creative_story_from_media")
        else:
            # Use chain-of-thought for complex multimodal tasks
            return prompt_patterns.chain_of_thought_multimodal(content_types, task_type)
    
    # Fallback to a general multimodal prompt
    return prompt_patterns.chain_of_thought_multimodal(content_types, task_type)