"""
Demonstration of New Campfires Features

This demo showcases the new features implemented in Campfires:
1. Role-aware orchestration with task decomposition
2. CampfireFactory for dynamic instantiation
3. PartyOrchestrator with execution topologies
4. ManifestLoader for YAML configuration
5. DefaultAuditor for task validation
6. Context path support for RAG
7. Torch rules engine for conditional processing

Run this demo to see how these components work together to create
a sophisticated task orchestration system.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import new Campfires features
from campfires.core import (
    # Orchestration
    TaskComplexity, RoleAwareOrchestrator, TaskDecomposer,
    DynamicRoleGenerator,
    
    # Factory
    CampfireFactory, CampfireTemplate,
    
    # Party orchestration
    PartyOrchestrator, ExecutionTopology,
    
    # Configuration
    ManifestLoader, CampfireManifest,
    
    # Validation
    DefaultAuditor, TaskRequirement,
    
    # Context management
    ContextPathManager, ContextType,
    
    # Rules engine
    TorchRulesEngine, create_simple_rule, create_routing_rule,
    RuleType, OperatorType, ActionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewFeaturesDemo:
    """Comprehensive demo of new Campfires features."""
    
    def __init__(self):
        """Initialize the demo."""
        self.orchestrator = None
        self.factory = None
        self.party_orchestrator = None
        self.manifest_loader = None
        self.auditor = None
        self.context_manager = None
        self.rules_engine = None
    
    async def run_demo(self):
        """Run the complete demonstration."""
        logger.info("🔥 Starting Campfires New Features Demo 🔥")
        
        try:
            # Initialize components
            await self.setup_components()
            
            # Demo 1: Role-aware orchestration
            await self.demo_role_aware_orchestration()
            
            # Demo 2: CampfireFactory
            await self.demo_campfire_factory()
            
            # Demo 3: PartyOrchestrator
            await self.demo_party_orchestrator()
            
            # Demo 4: ManifestLoader
            await self.demo_manifest_loader()
            
            # Demo 5: DefaultAuditor
            await self.demo_default_auditor()
            
            # Demo 6: Context path support
            await self.demo_context_path_support()
            
            # Demo 7: Torch rules engine
            await self.demo_torch_rules_engine()
            
            # Demo 8: Integrated workflow
            await self.demo_integrated_workflow()
            
            logger.info("✅ Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def setup_components(self):
        """Initialize all components."""
        logger.info("🔧 Setting up components...")
        
        # Initialize orchestrator
        self.orchestrator = RoleAwareOrchestrator()
        
        # Initialize factory
        self.factory = CampfireFactory()
        
        # Initialize party orchestrator
        self.party_orchestrator = PartyOrchestrator()
        
        # Initialize manifest loader
        self.manifest_loader = ManifestLoader()
        
        # Initialize auditor
        self.auditor = DefaultAuditor()
        
        # Initialize context manager
        self.context_manager = ContextPathManager()
        
        # Initialize rules engine
        self.rules_engine = TorchRulesEngine()
        
        logger.info("✅ Components initialized")
    
    async def demo_role_aware_orchestration(self):
        """Demonstrate role-aware orchestration."""
        logger.info("\n📋 Demo 1: Role-Aware Orchestration")
        
        # Define a complex task
        task_description = """
        Create a comprehensive data analysis pipeline that:
        1. Collects data from multiple sources
        2. Cleans and preprocesses the data
        3. Performs statistical analysis
        4. Generates visualizations
        5. Creates a summary report
        """
        
        # Decompose the task
        decomposition_result = await self.orchestrator.decompose_task(
            task_description=task_description,
            complexity_hint=TaskComplexity.HIGH,
            context={"domain": "data_analysis", "urgency": "medium"}
        )
        
        logger.info(f"Task decomposed into {len(decomposition_result.subtasks)} subtasks:")
        for i, subtask in enumerate(decomposition_result.subtasks, 1):
            logger.info(f"  {i}. {subtask.title} (complexity: {subtask.complexity.value})")
            logger.info(f"     Required role: {subtask.required_role.name}")
        
        # Generate dynamic roles
        for subtask in decomposition_result.subtasks:
            role_spec = await self.orchestrator.role_generator.generate_role_specification(
                subtask.required_role,
                {"task_context": subtask.description}
            )
            logger.info(f"Generated role spec for {subtask.required_role.name}:")
            logger.info(f"  Skills: {', '.join(role_spec.required_skills)}")
            logger.info(f"  Tools: {', '.join(role_spec.required_tools)}")
    
    async def demo_campfire_factory(self):
        """Demonstrate CampfireFactory."""
        logger.info("\n🏭 Demo 2: CampfireFactory")
        
        # Create a template for data analysis campfires
        template = CampfireTemplate(
            name="data_analysis_template",
            description="Template for data analysis tasks",
            base_config={
                "model": "openrouter/free",
                "temperature": 0.7,
                "max_tokens": 2000
            },
            required_roles=["data_analyst", "statistician"],
            resource_limits={
                "max_memory_mb": 1024,
                "max_execution_time_seconds": 300
            }
        )
        
        # Register the template
        self.factory.register_template(template)
        logger.info(f"Registered template: {template.name}")
        
        # Create campfire instances
        instance1 = await self.factory.create_campfire(
            template_name="data_analysis_template",
            instance_id="analysis_001",
            config_overrides={"temperature": 0.5}
        )
        
        instance2 = await self.factory.create_campfire(
            template_name="data_analysis_template",
            instance_id="analysis_002",
            config_overrides={"max_tokens": 1500}
        )
        
        logger.info(f"Created campfire instances:")
        logger.info(f"  - {instance1.instance_id} (status: {instance1.status.value})")
        logger.info(f"  - {instance2.instance_id} (status: {instance2.status.value})")
        
        # List active instances
        active_instances = self.factory.list_instances(status_filter="active")
        logger.info(f"Active instances: {len(active_instances)}")
    
    async def demo_party_orchestrator(self):
        """Demonstrate PartyOrchestrator."""
        logger.info("\n🎉 Demo 3: PartyOrchestrator")
        
        # Create a party for parallel data processing
        party_config = {
            "party_id": "data_processing_party",
            "description": "Parallel data processing workflow",
            "topology": ExecutionTopology.PARALLEL,
            "max_concurrent_tasks": 3
        }
        
        # Initialize party
        await self.party_orchestrator.initialize_party(party_config)
        
        # Add tasks to the party
        tasks = [
            {
                "task_id": "collect_data",
                "description": "Collect data from API",
                "campfire_template": "data_collector",
                "priority": 1
            },
            {
                "task_id": "clean_data",
                "description": "Clean and preprocess data",
                "campfire_template": "data_cleaner",
                "priority": 2,
                "dependencies": ["collect_data"]
            },
            {
                "task_id": "analyze_data",
                "description": "Perform statistical analysis",
                "campfire_template": "data_analyst",
                "priority": 3,
                "dependencies": ["clean_data"]
            }
        ]
        
        for task in tasks:
            await self.party_orchestrator.add_task_to_party(
                party_config["party_id"], 
                task
            )
        
        logger.info(f"Added {len(tasks)} tasks to party")
        
        # Execute the party (simulation)
        logger.info("Executing party with parallel topology...")
        execution_result = await self.party_orchestrator.execute_party(
            party_config["party_id"]
        )
        
        logger.info(f"Party execution completed:")
        logger.info(f"  - Success: {execution_result.success}")
        logger.info(f"  - Tasks completed: {len(execution_result.completed_tasks)}")
        logger.info(f"  - Execution time: {execution_result.execution_time_seconds:.2f}s")
    
    async def demo_manifest_loader(self):
        """Demonstrate ManifestLoader."""
        logger.info("\n📄 Demo 4: ManifestLoader")
        
        # Create a sample YAML manifest
        manifest_yaml = """
        version: "1.0"
        kind: "CampfireManifest"
        metadata:
          name: "text_analysis_campfire"
          description: "Campfire for text analysis tasks"
          
        spec:
          # Base configuration (like FROM in Dockerfile)
          from: "campfires/base:latest"
          
          # Environment variables (like ENV in Dockerfile)
          env:
            MODEL_NAME: "openrouter/free"
            TEMPERATURE: "0.7"
            MAX_TOKENS: "2000"
          
          # Copy configuration files (like COPY in Dockerfile)
          copy:
            - source: "./configs/analysis.yaml"
              dest: "/app/config/"
          
          # Run setup commands (like RUN in Dockerfile)
          run:
            - "pip install nltk spacy"
            - "python -m spacy download en_core_web_sm"
          
          # Expose capabilities (like EXPOSE in Dockerfile)
          expose:
            - capability: "text_analysis"
              port: 8080
          
          # Mount volumes (like VOLUME in Dockerfile)
          volume:
            - "/app/data"
            - "/app/models"
          
          # Set working directory (like WORKDIR in Dockerfile)
          workdir: "/app"
          
          # Default command (like CMD in Dockerfile)
          cmd: ["python", "text_analyzer.py"]
          
          # Campfire-specific configuration
          campers:
            - name: "text_processor"
              role: "text_analysis"
              skills: ["nlp", "sentiment_analysis", "entity_extraction"]
            - name: "report_generator"
              role: "reporting"
              skills: ["visualization", "document_generation"]
        """
        
        # Load and validate manifest
        try:
            manifest = self.manifest_loader.load_from_yaml(manifest_yaml)
            logger.info(f"Loaded manifest: {manifest.metadata['name']}")
            logger.info(f"  - Kind: {manifest.kind}")
            logger.info(f"  - Version: {manifest.version}")
            logger.info(f"  - Campers: {len(manifest.spec.get('campers', []))}")
            
            # Validate manifest
            validation_result = self.manifest_loader.validate_manifest(manifest)
            if validation_result.is_valid:
                logger.info("✅ Manifest validation passed")
            else:
                logger.warning(f"⚠️ Manifest validation issues: {len(validation_result.errors)}")
                
        except Exception as e:
            logger.error(f"❌ Manifest loading failed: {e}")
    
    async def demo_default_auditor(self):
        """Demonstrate DefaultAuditor."""
        logger.info("\n🔍 Demo 5: DefaultAuditor")
        
        # Define task requirements
        requirements = [
            TaskRequirement(
                id="req_001",
                description="Must process at least 1000 records",
                category="performance",
                priority="high",
                acceptance_criteria=["record_count >= 1000"]
            ),
            TaskRequirement(
                id="req_002", 
                description="Must complete within 5 minutes",
                category="performance",
                priority="medium",
                acceptance_criteria=["execution_time <= 300"]
            ),
            TaskRequirement(
                id="req_003",
                description="Must achieve 95% accuracy",
                category="quality",
                priority="high",
                acceptance_criteria=["accuracy >= 0.95"]
            )
        ]
        
        # Simulate task solution
        solution_data = {
            "record_count": 1250,
            "execution_time": 240,
            "accuracy": 0.97,
            "error_rate": 0.03,
            "memory_usage_mb": 512
        }
        
        # Audit the solution
        audit_result = await self.auditor.audit_task_solution(
            task_description="Process customer data and generate insights",
            requirements=requirements,
            solution_data=solution_data,
            context={"domain": "data_processing", "environment": "production"}
        )
        
        logger.info(f"Audit completed:")
        logger.info(f"  - Overall score: {audit_result.overall_score:.2f}")
        logger.info(f"  - Requirements met: {audit_result.requirements_met}/{len(requirements)}")
        logger.info(f"  - Issues found: {len(audit_result.issues)}")
        
        for issue in audit_result.issues:
            logger.info(f"    - {issue.severity.value}: {issue.description}")
    
    async def demo_context_path_support(self):
        """Demonstrate context path support."""
        logger.info("\n🗂️ Demo 6: Context Path Support")
        
        # Create context hierarchy
        await self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights",
            ContextType.PROJECT
        )
        
        await self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights/datasets",
            ContextType.DATASET
        )
        
        await self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights/models",
            ContextType.MODEL
        )
        
        # Add context items
        await self.context_manager.add_context_item(
            "projects/data_analysis/customer_insights/datasets",
            "customer_data.csv",
            {
                "description": "Customer transaction data",
                "size_mb": 150,
                "records": 50000,
                "last_updated": "2024-01-15"
            }
        )
        
        await self.context_manager.add_context_item(
            "projects/data_analysis/customer_insights/models",
            "churn_prediction_model.pkl",
            {
                "description": "Customer churn prediction model",
                "accuracy": 0.94,
                "model_type": "random_forest",
                "trained_on": "2024-01-10"
            }
        )
        
        # Query context
        query_result = await self.context_manager.query_context(
            path_pattern="projects/data_analysis/**",
            context_types=[ContextType.DATASET, ContextType.MODEL],
            filters={"size_mb": {"$gt": 100}}
        )
        
        logger.info(f"Context query results:")
        logger.info(f"  - Paths found: {len(query_result.matching_paths)}")
        logger.info(f"  - Items found: {len(query_result.context_items)}")
        
        for item in query_result.context_items:
            logger.info(f"    - {item.path}/{item.name}: {item.metadata.get('description', 'No description')}")
    
    async def demo_torch_rules_engine(self):
        """Demonstrate Torch rules engine."""
        logger.info("\n⚡ Demo 7: Torch Rules Engine")
        
        # Create routing rules
        priority_rule = create_simple_rule(
            rule_id="priority_routing",
            name="Priority-based routing",
            field="priority",
            operator="eq",
            value="high",
            action_type="route_to",
            action_target="high_priority_queue"
        )
        
        complexity_rule = create_simple_rule(
            rule_id="complexity_routing",
            name="Complexity-based routing",
            field="complexity",
            operator="gt",
            value=7,
            action_type="route_to",
            action_target="expert_queue"
        )
        
        # Add rules to engine
        self.rules_engine.add_rule(priority_rule)
        self.rules_engine.add_rule(complexity_rule)
        
        # Create test data
        test_cases = [
            {"priority": "high", "complexity": 5, "task_type": "analysis"},
            {"priority": "medium", "complexity": 8, "task_type": "modeling"},
            {"priority": "low", "complexity": 3, "task_type": "reporting"}
        ]
        
        # Execute rules against test data
        for i, test_data in enumerate(test_cases, 1):
            from campfires.core.torch_rules import RuleExecutionContext
            
            context = RuleExecutionContext(
                data=test_data,
                execution_id=f"test_{i}",
                source="demo"
            )
            
            results = await self.rules_engine.execute_rules(context)
            
            logger.info(f"Test case {i}: {test_data}")
            for result in results:
                if result.routing_decision:
                    logger.info(f"  → Routed to: {result.routing_decision}")
                else:
                    logger.info(f"  → No routing decision")
        
        # Show engine metrics
        metrics = self.rules_engine.get_metrics()
        logger.info(f"Rules engine metrics:")
        logger.info(f"  - Total rules: {metrics['rules_count']}")
        logger.info(f"  - Active rules: {metrics['active_rules_count']}")
        logger.info(f"  - Total executions: {metrics['total_executions']}")
    
    async def demo_integrated_workflow(self):
        """Demonstrate integrated workflow using all features."""
        logger.info("\n🔄 Demo 8: Integrated Workflow")
        
        # Scenario: Automated data science pipeline
        logger.info("Scenario: Automated data science pipeline")
        
        # 1. Use rules engine to classify incoming request
        from campfires.core.torch_rules import RuleExecutionContext
        
        request_data = {
            "data_size_gb": 5.2,
            "complexity": "high",
            "deadline_hours": 24,
            "domain": "finance"
        }
        
        context = RuleExecutionContext(
            data=request_data,
            execution_id="pipeline_001",
            source="api_request"
        )
        
        # Create classification rule
        size_rule = create_simple_rule(
            rule_id="data_size_classification",
            name="Classify by data size",
            field="data_size_gb",
            operator="gt",
            value=5.0,
            action_type="route_to",
            action_target="big_data_pipeline"
        )
        
        self.rules_engine.add_rule(size_rule)
        classification_results = await self.rules_engine.execute_rules(context)
        
        pipeline_type = "standard_pipeline"
        for result in classification_results:
            if result.routing_decision:
                pipeline_type = result.routing_decision
                break
        
        logger.info(f"1. Request classified → {pipeline_type}")
        
        # 2. Use orchestrator to decompose the task
        task_description = f"""
        Analyze financial data ({request_data['data_size_gb']}GB) to:
        - Detect anomalies in transactions
        - Predict market trends
        - Generate risk assessment report
        Deadline: {request_data['deadline_hours']} hours
        """
        
        decomposition = await self.orchestrator.decompose_task(
            task_description=task_description,
            complexity_hint=TaskComplexity.HIGH,
            context=request_data
        )
        
        logger.info(f"2. Task decomposed → {len(decomposition.subtasks)} subtasks")
        
        # 3. Use factory to create specialized campfires
        analysis_template = CampfireTemplate(
            name="financial_analysis_template",
            description="Template for financial data analysis",
            base_config={"model": "openrouter/free", "temperature": 0.3},
            required_roles=["financial_analyst", "data_scientist"],
            resource_limits={"max_memory_mb": 2048}
        )
        
        self.factory.register_template(analysis_template)
        
        campfire_instances = []
        for i, subtask in enumerate(decomposition.subtasks):
            instance = await self.factory.create_campfire(
                template_name="financial_analysis_template",
                instance_id=f"finance_analysis_{i+1}",
                config_overrides={"task_focus": subtask.title}
            )
            campfire_instances.append(instance)
        
        logger.info(f"3. Created {len(campfire_instances)} specialized campfires")
        
        # 4. Use party orchestrator to coordinate execution
        party_config = {
            "party_id": "financial_analysis_party",
            "description": "Financial data analysis workflow",
            "topology": ExecutionTopology.HIERARCHICAL,
            "max_concurrent_tasks": 2
        }
        
        await self.party_orchestrator.initialize_party(party_config)
        
        for i, (subtask, instance) in enumerate(zip(decomposition.subtasks, campfire_instances)):
            task_config = {
                "task_id": f"subtask_{i+1}",
                "description": subtask.description,
                "campfire_instance": instance.instance_id,
                "priority": 3 - i  # Higher priority for earlier tasks
            }
            await self.party_orchestrator.add_task_to_party(
                party_config["party_id"],
                task_config
            )
        
        logger.info("4. Party orchestrator configured for hierarchical execution")
        
        # 5. Use auditor to validate requirements
        requirements = [
            TaskRequirement(
                id="deadline_req",
                description=f"Must complete within {request_data['deadline_hours']} hours",
                category="performance",
                priority="high",
                acceptance_criteria=[f"execution_time <= {request_data['deadline_hours'] * 3600}"]
            ),
            TaskRequirement(
                id="accuracy_req",
                description="Must achieve high accuracy in predictions",
                category="quality", 
                priority="high",
                acceptance_criteria=["prediction_accuracy >= 0.90"]
            )
        ]
        
        logger.info(f"5. Defined {len(requirements)} validation requirements")
        
        # 6. Use context manager to organize results
        await self.context_manager.create_context_path(
            f"projects/financial_analysis/{context.execution_id}",
            ContextType.PROJECT
        )
        
        await self.context_manager.add_context_item(
            f"projects/financial_analysis/{context.execution_id}",
            "analysis_config.json",
            {
                "pipeline_type": pipeline_type,
                "subtasks_count": len(decomposition.subtasks),
                "campfires_created": len(campfire_instances),
                "requirements": [req.id for req in requirements]
            }
        )
        
        logger.info("6. Context organized for result tracking")
        
        # 7. Simulate execution and final audit
        simulated_results = {
            "execution_time": 18000,  # 5 hours in seconds
            "prediction_accuracy": 0.93,
            "anomalies_detected": 47,
            "risk_score": 0.23,
            "data_processed_gb": request_data['data_size_gb']
        }
        
        final_audit = await self.auditor.audit_task_solution(
            task_description=task_description,
            requirements=requirements,
            solution_data=simulated_results,
            context=request_data
        )
        
        logger.info(f"7. Final audit completed:")
        logger.info(f"   - Overall score: {final_audit.overall_score:.2f}")
        logger.info(f"   - Requirements met: {final_audit.requirements_met}/{len(requirements)}")
        logger.info(f"   - Success: {'✅' if final_audit.overall_score >= 0.8 else '❌'}")
        
        logger.info("\n🎯 Integrated workflow completed successfully!")
        logger.info("All new features demonstrated working together in harmony.")


async def main():
    """Run the demonstration."""
    demo = NewFeaturesDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())