"""
DefaultAuditor for RAG-prompted task validation.

This module provides an auditing system that validates whether solutions
fulfill task requirements using RAG (Retrieval-Augmented Generation) prompting.
The auditor focuses on task requirement validation and solution assessment.
"""

import json5
import json
import logging
import re
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .graph_store import GraphStore, Node, Edge

# Optional imports - will be used if available
try:
    from ..zeitgeist.zeitgeist_engine import ZeitgeistEngine
except ImportError:
    ZeitgeistEngine = None


logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during auditing."""
    severity: ValidationSeverity
    category: str
    description: str
    suggestion: str
    location: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    task_id: str
    task_description: str
    solution_summary: str
    overall_result: ValidationResult
    confidence_score: float
    validation_timestamp: datetime
    issues: List[ValidationIssue] = field(default_factory=list)
    requirements_coverage: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRequirement:
    """Represents a specific task requirement."""
    id: str
    description: str
    priority: str  # critical, high, medium, low
    validation_criteria: List[str]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditContext:
    """Context information for auditing."""
    task_id: str
    task_description: str
    requirements: List[TaskRequirement]
    solution_data: Dict[str, Any]
    execution_context: Dict[str, Any] = field(default_factory=dict)
    historical_data: List[Dict[str, Any]] = field(default_factory=list)


class DefaultAuditor:
    """
    Default auditor implementation with RAG-prompted task validation.
    
    The auditor validates whether solutions fulfill task requirements by:
    1. Analyzing task requirements using RAG prompting
    2. Evaluating solution completeness and correctness
    3. Checking requirement coverage
    4. Providing detailed validation reports
    5. Suggesting improvements and corrections
    """
    
    def __init__(self, 
                 party_box: Any = None,
                 zeitgeist_engine: Any = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the default auditor.
        
        Args:
            party_box: Optional PartyBox instance for RAG operations
            zeitgeist_engine: Optional Zeitgeist engine for context
            config: Auditor configuration
        """
        self.party_box = party_box
        self.zeitgeist_engine = zeitgeist_engine
        self.config = config or {}
        # Graph sharing configuration
        self._graph_store: Optional[GraphStore] = None
        self._graph_enabled: bool = self.config.get('graph_enabled', True)
        self._topic_aliases: Dict[str, str] = {
            **{k.lower(): v for k, v in (self.config.get('topic_aliases') or {}).items()}
        }
        
        # Auditing configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_rag_context_length = self.config.get('max_rag_context_length', 4000)
        self.validation_model = self.config.get('validation_model', 'meta-llama/llama-3.2-3b-instruct:free')
        self._rag_document_path: Optional[str] = None
        self._rag_content: Optional[str] = None
        self._load_rag_document()

        # Validation templates
        self._validation_prompts = {
            'requirement_analysis': """
            Analyze the following task requirements and solution to determine if the solution fulfills the requirements.
            
            Task Description: {task_description}
            
            Requirements:
            {requirements_text}
            
            Solution Summary:
            {solution_summary}
            
            Solution Details:
            {solution_details}
            
            Please evaluate:
            1. Does the solution address all stated requirements?
            2. Are there any missing or incomplete aspects?
            3. What is the quality and completeness of the solution?
            4. Are there any potential issues or improvements needed?
            
            Return ONLY a strict JSON object with these exact keys:
            - overall_assessment: one of ["pass", "fail", "partial", "needs_review"].
            - confidence_score: number between 0.0 and 1.0.
            - requirements_coverage: object mapping requirement IDs to true/false.
            - issues: array of objects with keys [severity, category, description, suggestion].
              - severity: one of ["critical", "high", "medium", "low", "info"].
              - category: one of ["completeness", "correctness", "quality", "performance", "security", "usability", "maintainability", "general"].
            - recommendations: array of strings.
            - reasoning: string.
            
            Output rules:
            - Strict JSON only (no code fences, no comments, no extra text).
            - Use quoted keys and values where appropriate.
            - Do not include example placeholders.
            """,
            
            'solution_quality': """
            Evaluate the quality and correctness of this solution for the given task.
            
            Task: {task_description}
            Solution Summary: {solution_summary}
            Solution Details: {solution_details}
            Historical Context: {historical_context}
            
            Return ONLY a strict JSON object with these exact keys:
            - quality_score: number between 0.0 and 1.0 (not a string).
            - quality_issues: array of objects with keys [severity, category, description, suggestion].
              - severity: one of ["critical", "high", "medium", "low", "info"].
              - category: one of ["completeness", "correctness", "quality", "performance", "security", "usability", "maintainability", "general"].
            - quality_recommendations: array of strings.
            - reasoning: string explaining the assessment.
            
            Output rules:
            - Strict JSON only (no code fences, no comments, no extra text).
            - Valid JSON syntax (quoted keys, no trailing commas).
            - Ensure quality_score is a number in [0.0, 1.0].
            """,
            
            'requirement_coverage': """
            Check if the provided solution covers all the specified requirements.
            
            Requirements:
            {requirements_list}
            
            Solution:
            {solution_data}
            
            For each requirement, determine if it's:
            - Fully satisfied
            - Partially satisfied
            - Not addressed
            - Cannot be determined
            
            Provide specific evidence for your assessment.
            """
        }
        
        # Issue categorization
        self._issue_categories = {
            'completeness': 'Solution completeness and coverage',
            'correctness': 'Technical correctness and accuracy',
            'quality': 'Code/solution quality and best practices',
            'performance': 'Performance and efficiency concerns',
            'security': 'Security and safety considerations',
            'usability': 'User experience and usability',
            'maintainability': 'Code maintainability and documentation'
        }

    def set_graph_store(self, graph_store: GraphStore) -> None:
        """
        Inject a GraphStore for auditor publishing and retrieval.
        """
        self._graph_store = graph_store

    def _normalize_topic(self, topic: Optional[str]) -> str:
        """
        Normalize topic using alias mapping. Falls back to lowercased stripped text.
        """
        if not topic:
            return "general"
        t = str(topic).strip().lower()
        return self._topic_aliases.get(t, t)
    
    async def audit_task_solution(self, audit_context: AuditContext) -> ValidationReport:
        """
        Audit a task solution against its requirements.
        
        Args:
            audit_context: Context containing task, requirements, and solution data
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"Starting audit for task: {audit_context.task_id}")
        
        try:
            logger.debug(f"Preparing RAG context for task: {audit_context.task_id}")
            # Prepare RAG context
            rag_context = await self._prepare_rag_context(audit_context)
            logger.debug(f"RAG context prepared. Length: {len(rag_context) if rag_context else 0}")
            
            logger.debug(f"Performing requirement analysis for task: {audit_context.task_id}")
            # Perform requirement analysis
            requirement_analysis = await self._analyze_requirements(audit_context, rag_context)
            logger.debug(f"Requirement analysis completed for task: {audit_context.task_id}. Overall assessment: {requirement_analysis.get('overall_assessment')}")
            
            logger.debug(f"Evaluating solution quality for task: {audit_context.task_id}")
            # Evaluate solution quality
            quality_assessment = await self._evaluate_solution_quality(audit_context, rag_context)
            logger.debug(f"Solution quality evaluation completed for task: {audit_context.task_id}. Quality score: {quality_assessment.get('quality_score')}")
            
            logger.debug(f"Checking requirement coverage for task: {audit_context.task_id}")
            # Check requirement coverage
            coverage_analysis = await self._check_requirement_coverage(audit_context)
            logger.debug(f"Requirement coverage checked for task: {audit_context.task_id}. Covered requirements: {sum(coverage_analysis.values())}/{len(coverage_analysis)}")
            
            logger.debug(f"Generating comprehensive report for task: {audit_context.task_id}")
            # Generate comprehensive report
            report = self._generate_validation_report(
                audit_context,
                requirement_analysis,
                quality_assessment,
                coverage_analysis
            )
            
            logger.info(f"Audit completed for task {audit_context.task_id}: {report.overall_result.value}")
            return report
        except Exception as e:
            logger.error(f"Audit failed for task {audit_context.task_id}: {e}")
            return self._create_error_report(audit_context, str(e))

    async def _prepare_rag_context(self, audit_context: AuditContext) -> str:
        """
        Prepare RAG context for validation prompting.
        
        Args:
            audit_context: Audit context
            
        Returns:
            RAG context string
        """
        # Build context query
        context_query = f"""
        Task validation context for: {audit_context.task_description}
        Requirements: {[req.description for req in audit_context.requirements]}
        Solution type: {audit_context.solution_data.get('type', 'unknown')}
        """
        
        # Retrieve relevant context using PartyBox if available
        try:
            rag_results = []
            if self.party_box:
                rag_results = await self.party_box.search_context(
                    query=context_query,
                    max_results=5,
                    context_type='validation'
                )
            
            # Combine RAG results into context
            context_parts = []
            for result in rag_results:
                context_parts.append(f"Context: {result.get('content', '')}")
            
            # Add historical data if available
            if audit_context.historical_data:
                context_parts.append("Historical validation data:")
                for hist_item in audit_context.historical_data[-3:]:  # Last 3 items
                    context_parts.append(f"- {hist_item.get('summary', '')}")
            
            # Add Zeitgeist context if available
            if self.zeitgeist_engine:
                zeitgeist_context = await self.zeitgeist_engine.get_zeitgeist(
                    topic=context_query,
                    role="auditor",  # Default role for auditor
                    context=context_type, # Use context_type as context
                    search_types=None # No specific search types for now
                )
                if zeitgeist_context and not zeitgeist_context.get('error'):
                    context_parts.append(f"Zeitgeist context: {zeitgeist_context.get('summary', zeitgeist_context)}")
            
            # Combine and truncate if necessary
            full_context = "\n".join(context_parts)
            if len(full_context) > self.max_rag_context_length:
                full_context = full_context[:self.max_rag_context_length] + "..."
            
            return full_context
            
        except Exception as e:
            logger.warning(f"Failed to prepare RAG context: {e}")
            return "No additional context available"
    
    async def _analyze_requirements(self, 
                                  audit_context: AuditContext, 
                                  rag_context: str) -> Dict[str, Any]:
        """
        Analyze requirements using RAG-prompted validation.
        
        Args:
            audit_context: Audit context
            rag_context: RAG context for prompting
            
        Returns:
            Requirements analysis results
        """
        # Format requirements for analysis
        requirements_text = "\n".join([
            f"- {req.id}: {req.description} (Priority: {req.priority})"
            for req in audit_context.requirements
        ])
        
        # Prepare solution summary
        solution_summary = audit_context.solution_data.get('summary', 'No summary provided')
        solution_details = json.dumps(audit_context.solution_data, indent=2)
        
        # Create validation prompt
        prompt = self._validation_prompts['requirement_analysis'].format(
            task_description=audit_context.task_description,
            requirements_text=requirements_text,
            solution_summary=solution_summary,
            solution_details=solution_details
        )
        
        # Add RAG context
        full_prompt = f"{rag_context}\n\n{prompt}"
        
        try:
            # Use PartyBox for LLM interaction if available
            if self.party_box:
                response = await self.party_box.query_llm(
                    prompt=full_prompt,
                    model=self.validation_model,
                    max_tokens=1500,
                    temperature=0.1  # Low temperature for consistent validation
                )
                logger.debug(f"LLM response for solution quality evaluation: {response}")
                logger.debug(f"LLM response for requirement analysis: {response}")
            else:
                # Fallback when no party_box available
                response = '{"overall_assessment": "pass", "confidence_score": 0.8, "requirements_coverage": {}, "issues": [], "recommendations": []}'
            
            # Parse JSON response
            try:
                json_match = re.search(r'```json\n(.*?)```', response, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1)
                    analysis_result = json5.loads(json_string)
                else:
                    # If no JSON block is found, try to parse the whole response as JSON
                    # This handles cases where the LLM might not wrap the JSON in ```json```
                    analysis_result = json5.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("LLM response is not valid JSON. Using fallback parser.")
                analysis_result = self._parse_fallback_response(response)
            except Exception as e:
                logger.error(f"An unexpected error occurred during JSON parsing: {e}")
                analysis_result = {}
            # Normalize field names
            if isinstance(analysis_result, dict):
                if 'identified_issues' in analysis_result and 'issues' not in analysis_result:
                    analysis_result['issues'] = analysis_result.pop('identified_issues')
            return analysis_result

        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return {
                'overall_assessment': 'error',
                'confidence_score': 0.0,
                'requirements_coverage': {},
                'issues': [],
                'recommendations': [],
                'reasoning': f'Analysis failed: {e}'
            }
    
    async def _analyze_requirements(self, audit_context: AuditContext, rag_context: str) -> Dict[str, Any]:
        """Analyze requirements coverage and solution quality."""
        try:
            # Improved prompt with strict JSON instructions
            prompt = f"""{self._validation_prompts['requirement_analysis']}"""
            response = await self.llm.generate(prompt=prompt)
            
            # Add JSON validation with error logging
            try:
                analysis_result = json.loads(response)
                # Validate required fields
                required_fields = ['overall_assessment', 'confidence_score', 'requirements_coverage', 'issues', 'recommendations', 'reasoning']
                if not all(field in analysis_result for field in required_fields):
                    missing_fields = [field for field in required_fields if field not in analysis_result]
                    raise ValueError(f"Missing required fields in LLM response: {', '.join(missing_fields)}")
                
                # Validate overall_assessment enum values
                valid_overall_assessments = [e.value for e in ValidationResult]
                if analysis_result.get('overall_assessment') not in valid_overall_assessments:
                    raise ValueError(f"Invalid overall_assessment value: {analysis_result.get('overall_assessment')}. Must be one of {', '.join(valid_overall_assessments)}")

            except json.JSONDecodeError as e:
                logger.error(f"LLM response is not valid JSON: {response}. Error: {e}")
                analysis_result = {
                    'overall_assessment': 'error',
                    'confidence_score': 0.0,
                    'requirements_coverage': {},
                    'issues': [],
                    'recommendations': [],
                    'reasoning': f'LLM response was not valid JSON: {e}'
                }
            except ValueError as e:
                logger.error(f"LLM response validation failed: {e}. Response: {response}")
                analysis_result = {
                    'overall_assessment': 'error',
                    'confidence_score': 0.0,
                    'requirements_coverage': {},
                    'issues': [],
                    'recommendations': [],
                    'reasoning': f'LLM response validation failed: {e}'
                }
            # Normalize field names
            if isinstance(analysis_result, dict):
                if 'identified_issues' in analysis_result and 'issues' not in analysis_result:
                    analysis_result['issues'] = analysis_result.pop('identified_issues')
            return analysis_result

        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return {
                'overall_assessment': 'error',
                'confidence_score': 0.0,
                'requirements_coverage': {},
                'issues': [],
                'recommendations': [],
                'reasoning': f'Analysis failed: {e}'
            }
    
    async def _evaluate_solution_quality(self, 
                                       audit_context: AuditContext, 
                                       rag_context: str) -> Dict[str, Any]:
        """
        Evaluate solution quality using RAG prompting.
        
        Args:
            audit_context: Audit context
            rag_context: RAG context for prompting
            
        Returns:
            Quality assessment results
        """
        # Prepare historical context
        historical_context = ""
        if audit_context.historical_data:
            historical_context = "\n".join([
                f"Previous task: {item.get('task', '')} - Result: {item.get('result', '')}"
                for item in audit_context.historical_data[-3:]
            ])
        
        # Create quality evaluation prompt
        solution_details = json.dumps(audit_context.solution_data, indent=2)
        prompt = self._validation_prompts['solution_quality'].format(
            task_description=audit_context.task_description,
            solution_summary=audit_context.solution_data.get('summary', ''),
            solution_details=solution_details,
            historical_context=historical_context
        )
        
        # Add RAG context
        full_prompt = f"{rag_context}\n\n{prompt}"
        
        try:
            if self.party_box:
                response = await self.party_box.query_llm(
                    prompt=full_prompt,
                    model=self.validation_model,
                    max_tokens=1200,
                    temperature=0.2
                )
            else:
                # Fallback when no party_box available
                response = "Quality Score: 8/10\nCompleteness: High\nCorrectness: Good\nEfficiency: Adequate\nMaintainability: Good"
            
            # Extract quality metrics from response
            with open("raw_llm_response_debug.txt", "w") as f:
                f.write(response)
            print(f"Raw LLM response type: {type(response)}")
            print(f"Raw LLM response: {response}")
            json_match = re.search(r'```(?:json)?\n(.*?)```', response, re.DOTALL)
            if json_match:
                json_string = json_match.group(1).strip()
            else:
                # Extract the first JSON object found in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_string = json_match.group().strip()
                else:
                    logger.warning("No JSON object found in LLM response. Using fallback parser.")
                    return self._parse_fallback_response(response)
            
            try:
                parsed_response = json5.loads(json_string)
            except json.JSONDecodeError:
                # Fallback to strict JSON parsing if json5 fails
                try:
                    parsed_response = json.loads(json_string)
                except json.JSONDecodeError:
                    logger.warning("LLM response contains invalid JSON. Using fallback parser.")
                    return self._parse_fallback_response(response)

            logger.debug(f"Parsed LLM response for quality evaluation: {parsed_response}")
            return self._extract_quality_metrics(parsed_response)
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                'quality_score': 0.0,
                'quality_issues': [],
                'quality_recommendations': []
            }
    
    async def _check_requirement_coverage(self, audit_context: AuditContext) -> Dict[str, bool]:
        """
        Check coverage of individual requirements.
        
        Args:
            audit_context: Audit context
            
        Returns:
            Requirement coverage mapping
        """
        coverage = {}
        
        for requirement in audit_context.requirements:
            # Create specific coverage check prompt
            prompt = self._validation_prompts['requirement_coverage'].format(
                requirements_list=f"{requirement.id}: {requirement.description}",
                solution_data=json.dumps(audit_context.solution_data, indent=2)
            )
            
            try:
                if self.party_box:
                    response = await self.party_box.query_llm(
                        prompt=prompt,
                        model=self.validation_model,
                        max_tokens=500,
                        temperature=0.1
                    )
                else:
                    # Fallback when no party_box available
                    response = "COVERED: true"
                
                # Simple heuristic to determine coverage
                response_lower = response.lower()
                if 'fully satisfied' in response_lower or 'completely addressed' in response_lower:
                    coverage[requirement.id] = True
                elif 'not addressed' in response_lower or 'missing' in response_lower:
                    coverage[requirement.id] = False
                else:
                    # Partial or uncertain - mark as False for safety
                    coverage[requirement.id] = False
                    
            except Exception as e:
                logger.warning(f"Coverage check failed for requirement {requirement.id}: {e}")
                coverage[requirement.id] = False
        
        return coverage
    
    def _generate_validation_report(self, 
                                  audit_context: AuditContext,
                                  requirement_analysis: Dict[str, Any],
                                  quality_assessment: Dict[str, Any],
                                  coverage_analysis: Dict[str, bool]) -> ValidationReport:
        """
        Generate comprehensive validation report.
        
        Args:
            audit_context: Audit context
            requirement_analysis: Requirements analysis results
            quality_assessment: Quality assessment results
            coverage_analysis: Coverage analysis results
            
        Returns:
            Validation report
        """
        # Determine overall result
        overall_result = self._determine_overall_result(
            requirement_analysis,
            quality_assessment,
            coverage_analysis
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            requirement_analysis,
            quality_assessment,
            coverage_analysis
        )
        
        # Collect all issues
        issues = []
        
        # Add requirement analysis issues
        for issue_data in requirement_analysis.get('issues', []):
            issues.append(ValidationIssue(
                severity=ValidationSeverity(issue_data.get('severity', 'medium')),
                category=issue_data.get('category', 'general'),
                description=issue_data.get('description', ''),
                suggestion=issue_data.get('suggestion', '')
            ))
        
        # Add quality issues
        for issue_data in quality_assessment.get('quality_issues', []):
            issues.append(ValidationIssue(
                severity=ValidationSeverity(issue_data.get('severity', 'medium')),
                category='quality',
                description=issue_data.get('description', ''),
                suggestion=issue_data.get('suggestion', '')
            ))
        
        # Collect recommendations
        recommendations = []
        recommendations.extend(requirement_analysis.get('recommendations', []))
        recommendations.extend(quality_assessment.get('quality_recommendations', []))
        
        # Create report
        report = ValidationReport(
            task_id=audit_context.task_id,
            task_description=audit_context.task_description,
            solution_summary=audit_context.solution_data.get('summary', ''),
            overall_result=overall_result,
            confidence_score=confidence_score,
            validation_timestamp=datetime.now(),
            issues=issues,
            requirements_coverage=coverage_analysis,
            recommendations=recommendations,
            metadata={
                'auditor_version': '1.0',
                'validation_model': self.validation_model,
                'requirement_count': len(audit_context.requirements),
                'analysis_data': {
                    'requirement_analysis': requirement_analysis,
                    'quality_assessment': quality_assessment
                }
            }
        )
        
        return report
    
    def _determine_overall_result(self, 
                                requirement_analysis: Dict[str, Any],
                                quality_assessment: Dict[str, Any],
                                coverage_analysis: Dict[str, bool]) -> ValidationResult:
        """Determine overall validation result."""
        # Check requirement analysis result
        req_result = requirement_analysis.get('overall_assessment', 'error')
        
        # Check coverage percentage
        total_requirements = len(coverage_analysis)
        covered_requirements = sum(coverage_analysis.values())
        coverage_percentage = covered_requirements / total_requirements if total_requirements > 0 else 0
        
        # Check for critical issues
        critical_issues = any(
            issue.get('severity') == 'critical' 
            for issue in requirement_analysis.get('issues', [])
        )
        
        # Determine result
        if critical_issues:
            return ValidationResult.FAIL
        elif req_result == 'fail':
            return ValidationResult.FAIL
        elif req_result == 'pass' and coverage_percentage >= 0.9:
            return ValidationResult.PASS
        elif req_result == 'partial' or coverage_percentage >= 0.7:
            return ValidationResult.PARTIAL
        elif req_result == 'needs_review':
            return ValidationResult.NEEDS_REVIEW
        else:
            return ValidationResult.FAIL
    
    def _calculate_confidence_score(self, 
                                  requirement_analysis: Dict[str, Any],
                                  quality_assessment: Dict[str, Any],
                                  coverage_analysis: Dict[str, bool]) -> float:
        """Calculate overall confidence score."""
        # Base confidence from requirement analysis
        base_confidence = requirement_analysis.get('confidence_score', 0.5)
        
        # Coverage factor
        coverage_percentage = sum(coverage_analysis.values()) / len(coverage_analysis) if coverage_analysis else 0
        coverage_factor = coverage_percentage
        
        # Quality factor
        quality_score = quality_assessment.get('quality_score', 0.5)
        
        # Weighted average
        confidence = (base_confidence * 0.5) + (coverage_factor * 0.3) + (quality_score * 0.2)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _create_error_report(self, audit_context: AuditContext, error_message: str) -> ValidationReport:
        """Create error validation report."""
        return ValidationReport(
            task_id=audit_context.task_id,
            task_description=audit_context.task_description,
            solution_summary="Error during validation",
            overall_result=ValidationResult.ERROR,
            confidence_score=0.0,
            validation_timestamp=datetime.now(),
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category='system',
                description=f"Validation error: {error_message}",
                suggestion="Review audit configuration and try again"
            )],
            requirements_coverage={},
            recommendations=["Fix validation system error"],
            metadata={'error': error_message}
        )
    
    def _parse_fallback_response(self, response: str) -> Dict[str, Any]:
        """Parse response when JSON parsing fails."""
        # Simple fallback parsing
        result = {
            'overall_assessment': 'needs_review',
            'confidence_score': 0.5,
            'requirements_coverage': {},
            'issues': [],
            'recommendations': [],
            'reasoning': response
        }
        
        # Try to extract some information
        response_lower = response.lower()
        if 'pass' in response_lower and 'fail' not in response_lower:
            result['overall_assessment'] = 'pass'
            result['confidence_score'] = 0.7
        elif 'fail' in response_lower:
            result['overall_assessment'] = 'fail'
            result['confidence_score'] = 0.3
        
        return result
    
    def _extract_quality_metrics(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate quality metrics from response dictionary."""
        # Check if quality_score exists
        if "quality_score" not in response_dict:
            # Heuristic fallback for demos: use a small default score when evaluator returns minimal/empty JSON
            logger.info("LLM response missing 'quality_score'; applying heuristic fallback of 0.3.")
            return {
                "quality_score": 0.3,
                "quality_issues": response_dict.get('quality_issues', []),
                "quality_recommendations": response_dict.get('quality_recommendations', [])
            }
        
        # Validate and convert quality_score to float
        quality_score = response_dict["quality_score"]
        if isinstance(quality_score, str):
            try:
                quality_score = float(quality_score.strip())
            except ValueError:
                logger.error(f"Invalid quality_score string: {quality_score}")
                quality_score = 0.0
        elif not isinstance(quality_score, (int, float)):
            logger.error(f"Invalid quality_score type: {type(quality_score)}")
            quality_score = 0.0
        
        # Ensure score is within 0-1 range
        quality_score = max(0.0, min(1.0, quality_score))
        
        quality_issues = response_dict.get('quality_issues', [])
        quality_recommendations = response_dict.get('quality_recommendations', [])
        
        return {
            'quality_score': quality_score,
            'quality_issues': quality_issues,
            'quality_recommendations': quality_recommendations
        }
    
    def get_validation_summary(self, report: ValidationReport) -> str:
        """
        Get a human-readable summary of the validation report.
        
        Args:
            report: Validation report
            
        Returns:
            Summary string
        """
        summary_parts = [
            f"Task: {report.task_description}",
            f"Result: {report.overall_result.value.upper()}",
            f"Confidence: {report.confidence_score:.2f}",
            f"Requirements Coverage: {sum(report.requirements_coverage.values())}/{len(report.requirements_coverage)}"
        ]
        
        if report.issues:
            critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
            high_issues = [i for i in report.issues if i.severity == ValidationSeverity.HIGH]
            
            if critical_issues:
                summary_parts.append(f"Critical Issues: {len(critical_issues)}")
            if high_issues:
                summary_parts.append(f"High Priority Issues: {len(high_issues)}")
        
        if report.recommendations:
            summary_parts.append(f"Recommendations: {len(report.recommendations)}")
        
        return " | ".join(summary_parts)

    def set_rag_document_path(self, path: Optional[str]):
        """
        Sets a new RAG document path and reloads the document.
        If path is None, clears the current RAG document.
        """
        self._rag_document_path = path
        self._load_rag_document()

    def _load_rag_document(self):
        """
        Loads the RAG document content from the specified path.
        If path is None or file does not exist, clears the content.
        """
        if self._rag_document_path and os.path.exists(self._rag_document_path):
            try:
                with open(self._rag_document_path, 'r', encoding='utf-8') as f:
                    self._rag_content = f.read()
                logger.info(f"Loaded RAG document from {self._rag_document_path}")
            except Exception as e:
                logger.error(f"Error loading RAG document from {self._rag_document_path}: {e}")
                self._rag_content = None
        else:
            self._rag_content = None
            if self._rag_document_path:
                logger.warning(f"RAG document path {self._rag_document_path} is invalid or does not exist. Clearing RAG content.")

    async def _analyze_requirements_and_solution(self, audit_context: AuditContext) -> Dict[str, Any]:
        """
        Placeholder for analyzing requirements and solution.
        Subclasses should override this method.
        """
        logger.warning("'_analyze_requirements_and_solution' not implemented in DefaultAuditor. Returning empty dict.")
        return {}

    async def audit(
        self,
        audit_context: AuditContext
    ) -> ValidationReport:
        """
        Performs an audit of the solution against the task requirements.
        """
        try:
            logger.info(f"Starting audit for task: {audit_context.task_id}")
        
            # 1. Analyze requirements and solution
            requirement_analysis = await self._analyze_requirements_and_solution(audit_context)
            
            rag_context = None # Initialize rag_context to None

            # Evaluate solution quality
            quality_assessment = await self._evaluate_solution_quality(audit_context, rag_context)
            
            # Check requirement coverage
            coverage_analysis = await self._check_requirement_coverage(audit_context)
            
            # Generate comprehensive report
            report = self._generate_validation_report(
                audit_context,
                requirement_analysis,
                quality_assessment,
                coverage_analysis
            )

            # Publish the audit report to graph for collaboration
            try:
                await self._share_stage_report(
                    audit_context=audit_context,
                    stage=audit_context.execution_context.get('stage') or 'audit',
                    report=report,
                    attempt=audit_context.execution_context.get('attempt')
                )
            except Exception as ge:
                logger.debug(f"Graph publish skipped/failed: {ge}")
            
            logger.info(f"Audit completed for task {audit_context.task_id}: {report.overall_result.value}")
            return report
        except Exception as e:
            logger.error(f"Audit failed for task {audit_context.task_id}: {e}")
            return self._create_error_report(audit_context, str(e))

    async def _share_stage_report(
        self,
        audit_context: AuditContext,
        stage: str,
        report: 'ValidationReport',
        attempt: Optional[int] = None,
    ) -> None:
        """
        Share the auditor's stage report to the graph store as a finding.
        """
        if not self._graph_enabled or self._graph_store is None:
            return
        # Derive topic from task description; normalize
        topic = self._normalize_topic(audit_context.task_description or stage)
        task_id = f"task:{audit_context.task_id}"
        stage_id = f"stage:{(stage or 'audit').strip().lower()}"
        auditor_id = f"auditor:{self.__class__.__name__}"
        finding_id = f"finding:{audit_context.task_id}:{stage}:{int(report.validation_timestamp.timestamp())}"

        # Upsert nodes
        await self._graph_store.upsert_node(Node(id=auditor_id, label="Auditor", properties={"name": self.__class__.__name__}))
        await self._graph_store.upsert_node(Node(id=task_id, label="Task", properties={"task_id": audit_context.task_id, "topic": topic}))
        await self._graph_store.upsert_node(Node(id=stage_id, label="Stage", properties={"name": stage}))
        await self._graph_store.upsert_node(Node(
            id=finding_id,
            label="Finding",
            properties={
                "summary": self.get_validation_summary(report),
                "importance": float(report.confidence_score or 0),
                "tags": [stage, report.overall_result.value],
                "timestamp": report.validation_timestamp.isoformat(),
            }
        ))

        # Link nodes
        await self._graph_store.upsert_edge(Edge(src=auditor_id, dst=finding_id, type="CREATED", properties={"attempt": attempt or 1}))
        await self._graph_store.upsert_edge(Edge(src=task_id, dst=finding_id, type="HAS_FINDING", properties={}))
        await self._graph_store.upsert_edge(Edge(src=stage_id, dst=finding_id, type="HAS_FINDING", properties={}))

        # Publish CAMAS procedure and steps linked to this stage and finding
        try:
            procedure_id = f"procedure:{stage_id.split(':',1)[1]}"
            await self._graph_store.upsert_node(Node(id=procedure_id, label="Procedure", properties={
                "name": stage,
                "type": "CAMAS"
            }))
            await self._graph_store.upsert_edge(Edge(src=stage_id, dst=procedure_id, type="HAS_PROCEDURE", properties={}))

            step_name = f"attempt-{(attempt or 1)}"
            proc_step_id = f"procstep:{audit_context.task_id}:{stage}:{(attempt or 1)}"
            await self._graph_store.upsert_node(Node(id=proc_step_id, label="ProcedureStep", properties={
                "name": step_name,
                "actor": "Auditor",
                "start_ts": report.validation_timestamp.isoformat(),
                "status": report.overall_result.value,
                "importance": float(report.confidence_score or 0)
            }))
            await self._graph_store.upsert_edge(Edge(src=procedure_id, dst=proc_step_id, type="HAS_STEP", properties={"order_index": (attempt or 1)}))
            await self._graph_store.upsert_edge(Edge(src=proc_step_id, dst=finding_id, type="GENERATED_FINDING", properties={}))
        except Exception as e:
            # Don't break auditing if procedure publishing fails
            pass

        # Add detailed context for retrieval weighting
        await self._graph_store.add_context(finding_id, {
            "confidence": float(report.confidence_score or 0),
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "description": i.description,
                    "suggestion": i.suggestion,
                } for i in (report.issues or [])
            ],
            "recommendations": report.recommendations or [],
            "coverage": report.requirements_coverage or {},
        })
from .graph_store import GraphStore, Node, Edge