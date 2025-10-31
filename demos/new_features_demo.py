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

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import logging
logging.basicConfig(level=logging.DEBUG)

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import new Campfires features
from campfires.core import (
    # Orchestration
    TaskComplexity, RoleAwareOrchestrator, TaskDecomposer,
    DynamicRoleGenerator, Torch,
    
    # Factory
    CampfireFactory, CampfireTemplate,
    
    # Party orchestration
    PartyOrchestrator, ExecutionTopology,
    
    # Configuration
    ManifestLoader, CampfireManifest,
    
    # Validation
    DefaultAuditor, TaskRequirement, AuditContext,
    
    # Context management
    ContextPathManager, ContextType,
    
    # Rules engine
    TorchRulesEngine, create_simple_rule, create_routing_rule,
    RuleType, OperatorType, ActionType
)
from campfires.core.ollama import OllamaConfig, OllamaClient, OllamaMCPClient
from campfires.mcp.ollama_protocol import OllamaMCPProtocol
from campfires.party_box import LocalDriver
from campfires.core.torch_rules import TorchRulesEngine

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
            # await self.demo_role_aware_orchestration()
            
            # Demo 2: CampfireFactory
            # await self.demo_campfire_factory()
            
            # Demo 3: PartyOrchestrator
            # await self.demo_party_orchestrator()
            
            # Demo 4: ManifestLoader
            await self.demo_manifest_loader()
            
            # Demo 5: DefaultAuditor
            await self.demo_default_auditor()
            
            # Demo 6: Context path support
            # await self.demo_context_path_support()
            
            # Demo 7: Torch rules engine
            await self.demo_torch_rules_engine()
            
            # Demo 8: Integrated workflow
            # await self.demo_integrated_workflow()
            
            logger.info("✅ Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def setup_components(self):
        """Initialize all components."""
        logger.info("🔧 Setting up components...")
        
        # Initialize orchestrator
        orchestrator_config = {
            "ollama_base_url": "http://localhost:11434",
            "ollama_model": "gemma3:latest"
        }
        self.orchestrator = RoleAwareOrchestrator(config=orchestrator_config)
        
        # Initialize party orchestrator
        ollama_config = OllamaConfig(
            base_url=orchestrator_config["ollama_base_url"],
            model="gemma3:latest"
        )
        ollama_mcp_client = OllamaMCPClient(ollama_config)
        ollama_mcp_protocol = OllamaMCPProtocol(ollama_config=ollama_config, campfire_name="PartyOrchestrator")
        await ollama_mcp_protocol.start()

        # Initialize factory
        party_box = LocalDriver(ollama_client=ollama_mcp_client.client)
        self.factory = CampfireFactory(party_box=party_box, config=orchestrator_config)
        
        self.party_orchestrator = PartyOrchestrator(
            orchestrator=self.orchestrator,
            config=orchestrator_config,
            party_box=party_box,
            campfire_factory=self.factory,
            mcp_protocol=ollama_mcp_protocol
        )
        
        # Initialize ManifestLoader, DefaultAuditor, and ContextPathManager
        from campfires.zeitgeist.zeitgeist_engine import ZeitgeistEngine
        from campfires.zeitgeist.config import ZeitgeistConfig

        zeitgeist_config = ZeitgeistConfig(
            preferred_search_engines=["duckduckgo"],
            max_search_results=5,
            cache_ttl=60 * 60 # 1 hour in seconds
        )
        self.zeitgeist_engine = ZeitgeistEngine(config=zeitgeist_config)
        self.manifest_loader = ManifestLoader()
        self.auditor = DefaultAuditor(party_box=self.context_manager, zeitgeist_engine=self.zeitgeist_engine, config=None)
        self.context_manager = ContextPathManager()
        self.torch_rules_engine = TorchRulesEngine()
        
        # Define tasks as Torch objects (these will be implicitly handled by execute_complex_task)
        # The claims and metadata will be passed as context to the orchestrator.
        context_tasks = [
            {
                "claim": "Collect data from API",
                "source_campfire": "data_processing_party",
                "channel": "data_collection",
                "metadata": {
                    "task_id": "collect_data",
                    "campfire_template": "data_collector",
                    "priority": 1
                }
            },
            {
                "claim": "Clean and preprocess data",
                "source_campfire": "data_processing_party",
                "channel": "data_cleaning",
                "metadata": {
                    "task_id": "clean_data",
                    "campfire_template": "data_cleaner",
                    "priority": 2,
                    "dependencies": ["collect_data"]
                }
            },
            {
                "claim": "Perform statistical analysis",
                "source_campfire": "data_processing_party",
                "channel": "data_analysis",
                "metadata": {
                    "task_id": "analyze_data",
                    "campfire_template": "data_analyst",
                    "priority": 3,
                    "dependencies": ["clean_data"]
                }
            }
        ]

        # Define party configuration
        party_config = {
            "party_id": "financial_analysis_party",
            "description": "Financial data analysis workflow",
            "topology": ExecutionTopology.HIERARCHICAL,
            "max_concurrent_tasks": 2
        }

        # Create execution plan using execute_complex_task
        await self.party_orchestrator.start()
        plan_id = await self.party_orchestrator.execute_complex_task(
            task_description=party_config["description"],
            topology=party_config["topology"],
            priority=party_config.get("priority", 5), # Assuming a default priority if not in config
            context={"tasks": context_tasks} # Pass the task details as context
        )
        logger.info(f"Created execution plan: {plan_id}")



        # Wait for the plan to complete (or for a certain duration)
        await asyncio.sleep(10)  # Wait for 10 seconds for demonstration

        # Get status of the plan
        plan_status = self.party_orchestrator.get_execution_status(plan_id)
        logger.info(f"Execution plan {plan_id} status: {plan_status.get('status')}")
        logger.info(f"Completed tasks: {len([t for t in plan_status.get('task_executions', []) if t.get('status') == 'completed'])}")

        # Stop the orchestrator (optional, depending on demo flow)
        await self.party_orchestrator.stop()
        logger.info("PartyOrchestrator stopped.")
    
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
            MODEL_NAME: "ollama/gemma3:latest"
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
            # Create a temporary file for the manifest
            temp_manifest_path = Path("./temp_manifest.yaml")
            with open(temp_manifest_path, "w", encoding="utf-8") as f:
                f.write(manifest_yaml)

            manifest = self.manifest_loader.load_campfire_manifest(str(temp_manifest_path))
            logger.info(f"Loaded manifest: {manifest.name}")
            logger.info(f"  - Version: {manifest.version}")
            logger.info(f"  - Campers: {len(manifest.campers)}")
            
            # Validate manifest
            validation_result = self.manifest_loader.validate_manifest_file(str(temp_manifest_path), 'campfire')
            if validation_result:
                logger.info("✅ Manifest validation passed")
            else:
                logger.warning(f"⚠️ Manifest validation issues.")
                
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
        finally:
            # Clean up the temporary file
            if temp_manifest_path.exists():
                temp_manifest_path.unlink()
    
    async def demo_default_auditor(self):
        """Demonstrate DefaultAuditor."""
        logger.info("\n🔍 Demo 5: DefaultAuditor")
        
        # Define task requirements
        requirements = [
            TaskRequirement(
                id="req_001",
                description="Must process at least 1000 records",
                priority="high",
                validation_criteria=["record_count >= 1000"]
            ),
            TaskRequirement(
                id="req_002", 
                description="Must complete within 5 minutes",
                priority="medium",
                validation_criteria=["execution_time <= 300"]
            ),
            TaskRequirement(
                id="req_003",
                description="Must achieve 95% accuracy",
                priority="high",
                validation_criteria=["accuracy >= 0.95"]
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
        audit_context = AuditContext(
            task_id="demo_task_001",  # Assign a unique ID for the demo task
            task_description="Process customer data and generate insights",
            requirements=requirements,
            solution_data=solution_data,
            execution_context={"domain": "data_processing", "environment": "production"}
        )
        audit_result = await self.auditor.audit_task_solution(audit_context)
        
        logger.info(f"Audit completed:")
        logger.info(f"  - Overall score: {audit_result.confidence_score:.2f}")
        logger.info(f"  - Requirements met: {len(audit_result.requirements_coverage)}/{len(requirements)}")
        logger.info(f"  - Issues found: {len(audit_result.issues)}")
        
        for issue in audit_result.issues:
            logger.info(f"    - {issue.severity.value}: {issue.description}")
    
    async def demo_context_path_support(self):
        """Demonstrate context path support."""
        logger.info("\n🗂️ Demo 6: Context Path Support")
        
        # Create context hierarchy
        self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights"
        )
        
        self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights/datasets"
        )
        
        self.context_manager.create_context_path(
            "projects/data_analysis/customer_insights/models"
        )
        
        # Add context items
        

        
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
        self.torch_rules_engine.add_rule(priority_rule)
        self.torch_rules_engine.add_rule(complexity_rule)
        
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
            
            results = await self.torch_rules_engine.execute_rules(context)
            
            logger.info(f"Test case {i}: {test_data}")
            for result in results:
                if result.routing_decision:
                    logger.info(f"  → Routed to: {result.routing_decision}")
                else:
                    logger.info(f"  → No routing decision")
        
        # Show engine metrics
        metrics = self.torch_rules_engine.get_metrics()
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
        
        self.torch_rules_engine.add_rule(size_rule)
        classification_results = await self.torch_rules_engine.execute_rules(context)
        
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
        
        decomposition = await self.orchestrator.task_decomposer.decompose_task(
            torch=Torch(
                claim=task_description,
                source_campfire="new_features_demo",
                channel="integrated_workflow",
                metadata={
                    "context": request_data,
                    "complexity_hint": TaskComplexity.HIGHLY_COMPLEX.value
                }
            )
        )
        
        logger.info(f"2. Task decomposed → {len(decomposition)} subtasks")
        
        # 3. Use factory to create specialized campfires
        analysis_template = CampfireTemplate(
                name="financial_analysis_template",
                description="Template for financial data analysis",
                camper_types=["dynamic"],
                default_config={"llm_provider": "ollama", "ollama_model": "gemma3:latest", "temperature": 0.3},
                resource_requirements={'memory': 'low', 'cpu': 'low'},
                max_concurrent_tasks=3,
                timeout_minutes=30
            )
        
        self.factory.add_template(analysis_template)
        

        
        context_tasks = [
            {"id": subtask.id, "description": subtask.description} for subtask in decomposition
        ]

        # 4. Use party orchestrator to coordinate execution
        party_config = {
            "party_id": "financial_analysis_party",
            "description": "Financial data analysis workflow",
            "topology": ExecutionTopology.HIERARCHICAL,
            "max_concurrent_tasks": 2
        }
        
        plan_id = await self.party_orchestrator.execute_complex_task(
            task_description=party_config["description"],
            topology=party_config["topology"],
            priority=party_config.get("priority", 5),
            context={"tasks": context_tasks}
        )
        

        
        logger.info("4. Party orchestrator configured for hierarchical execution")
        
        # 5. Use auditor to validate requirements
        requirements = [
            TaskRequirement(
                id="deadline_req",
                description=f"Must complete within {request_data['deadline_hours']} hours",
                priority="high",
                validation_criteria=[f"execution_time <= {request_data['deadline_hours'] * 3600}"]
            ),
            TaskRequirement(
                id="accuracy_req",
                description="Must achieve high accuracy in predictions",
                priority="high",
                validation_criteria=["prediction_accuracy >= 0.90"]
            )
        ]
        
        logger.info(f"5. Defined {len(requirements)} validation requirements")
        
        # 6. Use context manager to organize results
        self.context_manager.create_context_path(
            f"projects/financial_analysis/{context.execution_id}"
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
            AuditContext(
                task_id="integrated_workflow_task",
                task_description=task_description,
                requirements=requirements,
                solution_data=simulated_results,
                execution_context=request_data
            )
        )
        
        logger.info(f"7. Final audit completed:")
        logger.info(f"   - Overall score: {final_audit.confidence_score:.2f}")
        logger.info(f"   - Requirements met: {sum(final_audit.requirements_coverage.values())}/{len(requirements)}")
        logger.info(f"   - Success: {'✅' if final_audit.confidence_score >= 0.8 else '❌'}")
        
        logger.info("\n🎯 Integrated workflow completed successfully!")
        logger.info("All new features demonstrated working together in harmony.")


async def main():
    """Run the demonstration."""
    demo = NewFeaturesDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())