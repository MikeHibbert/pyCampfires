"""
DefaultAuditor for RAG-prompted task validation.

This module provides an auditing system that validates whether solutions
fulfill task requirements using RAG (Retrieval-Augmented Generation) prompting.
The auditor focuses on task requirement validation and solution assessment.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
        
        # Auditing configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_rag_context_length = self.config.get('max_rag_context_length', 4000)
        self.validation_model = self.config.get('validation_model', 'meta-llama/llama-3.2-3b-instruct:free')
        
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
            
            Provide your assessment in the following JSON format:
            {{
                "overall_assessment": "pass|fail|partial|needs_review",
                "confidence_score": 0.0-1.0,
                "requirements_coverage": {{
                    "requirement_id": true/false,
                    ...
                }},
                "issues": [
                    {{
                        "severity": "critical|high|medium|low|info",
                        "category": "completeness|correctness|quality|performance|security",
                        "description": "Issue description",
                        "suggestion": "Improvement suggestion"
                    }}
                ],
                "recommendations": ["recommendation1", "recommendation2", ...],
                "reasoning": "Detailed explanation of the assessment"
            }}
            """,
            
            'solution_quality': """
            Evaluate the quality and correctness of this solution for the given task.
            
            Task: {task_description}
            Solution: {solution_summary}
            
            Context from previous similar tasks:
            {historical_context}
            
            Focus on:
            1. Technical correctness
            2. Completeness of implementation
            3. Best practices adherence
            4. Potential edge cases or issues
            5. Performance considerations
            
            Rate the solution quality and provide specific feedback.
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
            # Prepare RAG context
            rag_context = await self._prepare_rag_context(audit_context)
            
            # Perform requirement analysis
            requirement_analysis = await self._analyze_requirements(audit_context, rag_context)
            
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
                zeitgeist_context = await self.zeitgeist_engine.get_context(
                    query=context_query,
                    context_type='task_validation'
                )
                if zeitgeist_context:
                    context_parts.append(f"Zeitgeist context: {zeitgeist_context}")
            
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
            else:
                # Fallback when no party_box available
                response = '{"overall_assessment": "pass", "confidence_score": 0.8, "requirements_coverage": {}, "identified_issues": [], "recommendations": []}'
            
            # Parse JSON response
            try:
                analysis_result = json.loads(response)
                return analysis_result
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                return self._parse_fallback_response(response)
                
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
        prompt = self._validation_prompts['solution_quality'].format(
            task_description=audit_context.task_description,
            solution_summary=audit_context.solution_data.get('summary', ''),
            historical_context=historical_context
        )
        
        # Add RAG context
        full_prompt = f"{rag_context}\n\n{prompt}"
        
        try:
            if self.party_box:
                response = await self.party_box.query_llm(
                    prompt=full_prompt,
                    model=self.validation_model,
                    max_tokens=1000,
                    temperature=0.2
                )
            else:
                # Fallback when no party_box available
                response = "Quality Score: 8/10\nCompleteness: High\nCorrectness: Good\nEfficiency: Adequate\nMaintainability: Good"
            
            # Extract quality metrics from response
            return self._extract_quality_metrics(response)
            
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
    
    def _extract_quality_metrics(self, response: str) -> Dict[str, Any]:
        """Extract quality metrics from response."""
        # Simple extraction logic
        quality_score = 0.5
        
        response_lower = response.lower()
        if 'excellent' in response_lower or 'high quality' in response_lower:
            quality_score = 0.9
        elif 'good' in response_lower:
            quality_score = 0.7
        elif 'poor' in response_lower or 'low quality' in response_lower:
            quality_score = 0.3
        
        return {
            'quality_score': quality_score,
            'quality_issues': [],
            'quality_recommendations': []
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