"""
Tests for multimodal prompt functionality.
"""

import pytest
from unittest.mock import Mock, patch
from campfires.core.multimodal_prompts import (
    PromptType,
    PromptTemplate,
    MultimodalPromptLibrary,
    PromptEngineeringPatterns,
    MultimodalPromptBuilder,
    get_prompt_for_content_types
)


class TestPromptType:
    """Test PromptType enum."""
    
    def test_prompt_types_exist(self):
        """Test that all expected prompt types exist."""
        assert hasattr(PromptType, 'VISION_ANALYSIS')
        assert hasattr(PromptType, 'AUDIO_TRANSCRIPTION')
        assert hasattr(PromptType, 'MULTIMODAL_QA')
        assert hasattr(PromptType, 'CONTENT_DESCRIPTION')
        assert hasattr(PromptType, 'CONTENT_COMPARISON')
        assert hasattr(PromptType, 'TEXT_EXTRACTION')
        assert hasattr(PromptType, 'SENTIMENT_ANALYSIS')
        assert hasattr(PromptType, 'CONTENT_CLASSIFICATION')
        assert hasattr(PromptType, 'CREATIVE_GENERATION')
        assert hasattr(PromptType, 'TECHNICAL_ANALYSIS')


class TestPromptTemplate:
    """Test PromptTemplate class."""
    
    def test_prompt_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test_template",
            prompt_type=PromptType.VISION_ANALYSIS,
            template="Analyze this image: {content}",
            description="Test template for image analysis",
            variables=["content"],
            content_types=["image"]
        )
        
        assert template.name == "test_template"
        assert template.prompt_type == PromptType.VISION_ANALYSIS
        assert template.template == "Analyze this image: {content}"
        assert template.description == "Test template for image analysis"
        assert template.variables == ["content"]
        assert template.content_types == ["image"]
    
    def test_prompt_template_format(self):
        """Test formatting a prompt template."""
        template = PromptTemplate(
            name="test_template",
            prompt_type=PromptType.VISION_ANALYSIS,
            template="Analyze this {content_type}: {content}",
            variables=["content_type", "content"]
        )
        
        formatted = template.format(content_type="image", content="[IMAGE_DATA]")
        assert formatted == "Analyze this image: [IMAGE_DATA]"
    
    def test_prompt_template_format_missing_variable(self):
        """Test formatting with missing variable."""
        template = PromptTemplate(
            name="test_template",
            prompt_type=PromptType.VISION_ANALYSIS,
            template="Analyze this {content_type}: {content}",
            variables=["content_type", "content"]
        )
        
        with pytest.raises(KeyError):
            template.format(content_type="image")  # Missing 'content'
    
    def test_prompt_template_supports_content_type(self):
        """Test checking if template supports content type."""
        template = PromptTemplate(
            name="test_template",
            prompt_type=PromptType.VISION_ANALYSIS,
            template="Analyze this image",
            content_types=["image"]
        )
        
        assert template.supports_content_type("image") is True
        assert template.supports_content_type("audio") is False
    
    def test_prompt_template_supports_any_content_type(self):
        """Test template that supports any content type."""
        template = PromptTemplate(
            name="test_template",
            prompt_type=PromptType.CONTENT_DESCRIPTION,
            template="Describe this content",
            content_types=None  # Supports any type
        )
        
        assert template.supports_content_type("image") is True
        assert template.supports_content_type("audio") is True
        assert template.supports_content_type("video") is True


class TestMultimodalPromptLibrary:
    """Test MultimodalPromptLibrary class."""
    
    @pytest.fixture
    def library(self):
        """Create a prompt library instance."""
        return MultimodalPromptLibrary()
    
    def test_library_initialization(self, library):
        """Test library initialization with default templates."""
        assert len(library.templates) > 0
        assert "vision_analysis" in library.templates
        assert "audio_transcription" in library.templates
    
    def test_add_template(self, library):
        """Test adding a custom template."""
        template = PromptTemplate(
            name="custom_template",
            prompt_type=PromptType.CONTENT_DESCRIPTION,
            template="Custom template: {content}",
            variables=["content"]
        )
        
        library.add_template(template)
        assert "custom_template" in library.templates
        assert library.templates["custom_template"] == template
    
    def test_get_template_existing(self, library):
        """Test getting an existing template."""
        template = library.get_template("vision_analysis")
        assert template is not None
        assert template.name == "vision_analysis"
    
    def test_get_template_nonexistent(self, library):
        """Test getting a non-existent template."""
        template = library.get_template("nonexistent_template")
        assert template is None
    
    def test_get_templates_by_type(self, library):
        """Test getting templates by prompt type."""
        vision_templates = library.get_templates_by_type(PromptType.VISION_ANALYSIS)
        assert len(vision_templates) > 0
        assert all(t.prompt_type == PromptType.VISION_ANALYSIS for t in vision_templates)
    
    def test_get_templates_for_content_type(self, library):
        """Test getting templates for specific content type."""
        image_templates = library.get_templates_for_content_type("image")
        assert len(image_templates) > 0
        assert all(t.supports_content_type("image") for t in image_templates)
    
    def test_list_template_names(self, library):
        """Test listing all template names."""
        names = library.list_template_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "vision_analysis" in names


class TestPromptEngineeringPatterns:
    """Test PromptEngineeringPatterns class."""
    
    def test_chain_of_thought(self):
        """Test chain of thought pattern."""
        base_prompt = "Analyze this image"
        cot_prompt = PromptEngineeringPatterns.chain_of_thought(base_prompt)
        
        assert "step by step" in cot_prompt.lower()
        assert base_prompt in cot_prompt
    
    def test_few_shot_examples(self):
        """Test few-shot examples pattern."""
        base_prompt = "Classify this image"
        examples = [
            {"input": "Image of a cat", "output": "Animal: Cat"},
            {"input": "Image of a car", "output": "Vehicle: Car"}
        ]
        
        few_shot_prompt = PromptEngineeringPatterns.few_shot_examples(base_prompt, examples)
        
        assert "Image of a cat" in few_shot_prompt
        assert "Animal: Cat" in few_shot_prompt
        assert base_prompt in few_shot_prompt
    
    def test_role_based_prompting(self):
        """Test role-based prompting pattern."""
        base_prompt = "Analyze this medical image"
        role_prompt = PromptEngineeringPatterns.role_based_prompting(
            base_prompt, 
            "medical expert"
        )
        
        assert "medical expert" in role_prompt.lower()
        assert base_prompt in role_prompt
    
    def test_structured_output(self):
        """Test structured output pattern."""
        base_prompt = "Describe this image"
        structure = {
            "objects": "List of objects in the image",
            "colors": "Dominant colors",
            "mood": "Overall mood or atmosphere"
        }
        
        structured_prompt = PromptEngineeringPatterns.structured_output(base_prompt, structure)
        
        assert "objects" in structured_prompt
        assert "colors" in structured_prompt
        assert "mood" in structured_prompt
        assert base_prompt in structured_prompt
    
    def test_comparative_analysis(self):
        """Test comparative analysis pattern."""
        base_prompt = "Compare these images"
        comparative_prompt = PromptEngineeringPatterns.comparative_analysis(base_prompt)
        
        assert "compare" in comparative_prompt.lower() or "comparison" in comparative_prompt.lower()
        assert "similarities" in comparative_prompt.lower()
        assert "differences" in comparative_prompt.lower()
    
    def test_contextual_prompting(self):
        """Test contextual prompting pattern."""
        base_prompt = "Analyze this image"
        context = "This image was taken during a medical examination"
        
        contextual_prompt = PromptEngineeringPatterns.contextual_prompting(base_prompt, context)
        
        assert context in contextual_prompt
        assert base_prompt in contextual_prompt


class TestMultimodalPromptBuilder:
    """Test MultimodalPromptBuilder class."""
    
    @pytest.fixture
    def builder(self):
        """Create a prompt builder instance."""
        return MultimodalPromptBuilder()
    
    def test_builder_initialization(self, builder):
        """Test builder initialization."""
        assert builder.library is not None
        assert builder.patterns is not None
    
    def test_build_simple_prompt(self, builder):
        """Test building a simple prompt."""
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]"
        )
        
        assert prompt is not None
        assert "[IMAGE_DATA]" in prompt
    
    def test_build_prompt_with_variables(self, builder):
        """Test building prompt with custom variables."""
        prompt = builder.build_prompt(
            template_name="content_description",
            content="[IMAGE_DATA]",
            detail_level="detailed"
        )
        
        assert prompt is not None
        assert "[IMAGE_DATA]" in prompt
    
    def test_build_prompt_with_chain_of_thought(self, builder):
        """Test building prompt with chain of thought."""
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]",
            use_chain_of_thought=True
        )
        
        assert prompt is not None
        assert "step by step" in prompt.lower()
    
    def test_build_prompt_with_examples(self, builder):
        """Test building prompt with few-shot examples."""
        examples = [
            {"input": "Image of a dog", "output": "This is a domestic animal, specifically a dog."}
        ]
        
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]",
            examples=examples
        )
        
        assert prompt is not None
        assert "Image of a dog" in prompt
    
    def test_build_prompt_with_role(self, builder):
        """Test building prompt with role-based prompting."""
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]",
            role="art critic"
        )
        
        assert prompt is not None
        assert "art critic" in prompt.lower()
    
    def test_build_prompt_with_structure(self, builder):
        """Test building prompt with structured output."""
        structure = {
            "description": "Brief description of the image",
            "objects": "List of objects detected"
        }
        
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]",
            output_structure=structure
        )
        
        assert prompt is not None
        assert "description" in prompt
        assert "objects" in prompt
    
    def test_build_prompt_with_context(self, builder):
        """Test building prompt with context."""
        context = "This image is from a security camera"
        
        prompt = builder.build_prompt(
            template_name="vision_analysis",
            content="[IMAGE_DATA]",
            context=context
        )
        
        assert prompt is not None
        assert context in prompt
    
    def test_build_prompt_nonexistent_template(self, builder):
        """Test building prompt with non-existent template."""
        with pytest.raises(ValueError):
            builder.build_prompt(
                template_name="nonexistent_template",
                content="[IMAGE_DATA]"
            )
    
    def test_build_multimodal_prompt(self, builder):
        """Test building a multimodal prompt with multiple content types."""
        contents = {
            "image": "[IMAGE_DATA]",
            "text": "What do you see in this image?"
        }
        
        prompt = builder.build_multimodal_prompt(
            template_name="multimodal_qa",
            contents=contents
        )
        
        assert prompt is not None
        assert "[IMAGE_DATA]" in prompt
        assert "What do you see in this image?" in prompt


class TestGetPromptForContentTypes:
    """Test get_prompt_for_content_types function."""
    
    def test_get_prompt_for_single_content_type(self):
        """Test getting prompt for single content type."""
        prompt = get_prompt_for_content_types(["image"], PromptType.VISION_ANALYSIS)
        
        assert prompt is not None
        assert isinstance(prompt, str)
    
    def test_get_prompt_for_multiple_content_types(self):
        """Test getting prompt for multiple content types."""
        prompt = get_prompt_for_content_types(
            ["image", "text"], 
            PromptType.MULTIMODAL_QA
        )
        
        assert prompt is not None
        assert isinstance(prompt, str)
    
    def test_get_prompt_for_unsupported_content_type(self):
        """Test getting prompt for unsupported content type."""
        prompt = get_prompt_for_content_types(
            ["unknown_type"], 
            PromptType.VISION_ANALYSIS
        )
        
        # Should return a generic prompt or None
        assert prompt is None or isinstance(prompt, str)
    
    def test_get_prompt_with_custom_variables(self):
        """Test getting prompt with custom variables."""
        prompt = get_prompt_for_content_types(
            ["image"], 
            PromptType.VISION_ANALYSIS,
            detail_level="high",
            focus="objects"
        )
        
        assert prompt is not None
        assert isinstance(prompt, str)


if __name__ == "__main__":
    pytest.main([__file__])