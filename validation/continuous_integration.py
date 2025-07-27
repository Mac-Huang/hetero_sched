"""
Continuous Integration for Research Experiments

This module implements R37: a comprehensive CI/CD framework specifically designed for 
machine learning research experiments in heterogeneous scheduling.

Key Features:
1. Automated experiment pipeline with version control integration
2. Research-specific testing including model performance regression tests
3. Distributed experiment execution across heterogeneous compute resources
4. Artifact management with automatic model versioning and dataset tracking
5. Performance monitoring and automated alerting for research metrics
6. Integration with containerized reproducibility framework

The system ensures that all research experiments are reproducible, tracked, and 
automatically validated across different hardware configurations.

Authors: HeteroSched Research Team
"""

import os
import yaml
import json
import docker
import asyncio
import hashlib
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import datetime
import logging
import git
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    REGRESSION = "regression"
    SMOKE = "smoke"

class ResourceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MIXED = "mixed"

@dataclass
class ExperimentConfig:
    """Configuration for a research experiment"""
    experiment_id: str
    name: str
    description: str
    script_path: str
    environment: Dict[str, Any]
    parameters: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    expected_outputs: List[str]
    performance_thresholds: Dict[str, float]
    timeout_minutes: int = 120
    retry_count: int = 2
    
@dataclass
class TestCase:
    """Represents a test case for CI pipeline"""
    test_id: str
    name: str
    test_type: TestType
    script_path: str
    expected_outputs: List[str]
    performance_metrics: Dict[str, float]
    timeout_minutes: int = 30
    dependencies: List[str] = field(default_factory=list)
    
@dataclass
class PipelineRun:
    """Represents a complete CI pipeline execution"""
    run_id: str
    commit_hash: str
    branch: str
    trigger_event: str
    timestamp: datetime.datetime
    status: ExperimentStatus
    experiments: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
@dataclass
class ComputeResource:
    """Represents available compute resources"""
    resource_id: str
    resource_type: ResourceType
    specifications: Dict[str, Any]
    availability_schedule: List[Dict[str, Any]]
    current_utilization: float = 0.0
    is_healthy: bool = True

class ContinuousIntegrationFramework:
    """
    Main CI framework for research experiments
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CI_Framework")
        
        # Initialize components
        self.experiment_manager = ExperimentManager(config)
        self.test_runner = TestRunner(config)
        self.resource_scheduler = ResourceScheduler(config)
        self.artifact_manager = ArtifactManager(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        # Pipeline state
        self.active_runs: Dict[str, PipelineRun] = {}
        self.run_history: List[PipelineRun] = []
        
        # Setup directories
        self.setup_workspace()
        
    def setup_workspace(self):
        """Setup CI workspace and directory structure"""
        workspace_root = Path(self.config.get("workspace_root", "ci_workspace"))
        
        # Create directory structure
        directories = [
            "experiments", "tests", "artifacts", "logs", 
            "configs", "scripts", "results", "reports"
        ]
        
        for dir_name in directories:
            (workspace_root / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.workspace_root = workspace_root
        self.logger.info(f"CI workspace initialized at {workspace_root}")
    
    async def trigger_pipeline(self, trigger_event: str, 
                             branch: str = "main") -> str:
        """Trigger a complete CI pipeline run"""
        
        # Get current commit info
        repo = git.Repo(".")
        commit_hash = repo.head.commit.hexsha
        
        # Create pipeline run
        run_id = self.generate_run_id(commit_hash, trigger_event)
        pipeline_run = PipelineRun(
            run_id=run_id,
            commit_hash=commit_hash,
            branch=branch,
            trigger_event=trigger_event,
            timestamp=datetime.datetime.now(),
            status=ExperimentStatus.PENDING
        )
        
        self.active_runs[run_id] = pipeline_run
        self.logger.info(f"Started pipeline run {run_id} for commit {commit_hash[:8]}")
        
        try:
            # Execute pipeline stages
            await self.execute_pipeline_stages(pipeline_run)
            pipeline_run.status = ExperimentStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Pipeline run {run_id} failed: {e}")
            pipeline_run.status = ExperimentStatus.FAILED
            raise
        
        finally:
            # Cleanup and finalize
            pipeline_run.duration_seconds = (
                datetime.datetime.now() - pipeline_run.timestamp
            ).total_seconds()
            
            self.run_history.append(pipeline_run)
            del self.active_runs[run_id]
        
        return run_id
    
    async def execute_pipeline_stages(self, pipeline_run: PipelineRun):
        """Execute all stages of the CI pipeline"""
        
        self.logger.info(f"Executing pipeline stages for run {pipeline_run.run_id}")
        
        # Stage 1: Environment setup and validation
        await self.stage_environment_setup(pipeline_run)
        
        # Stage 2: Unit and integration tests
        await self.stage_run_tests(pipeline_run)
        
        # Stage 3: Performance regression tests
        await self.stage_performance_tests(pipeline_run)
        
        # Stage 4: Research experiments
        await self.stage_run_experiments(pipeline_run)
        
        # Stage 5: Results validation and artifact generation
        await self.stage_validate_results(pipeline_run)
        
        # Stage 6: Performance monitoring and alerting
        await self.stage_performance_monitoring(pipeline_run)
        
        # Stage 7: Report generation
        await self.stage_generate_reports(pipeline_run)
    
    async def stage_environment_setup(self, pipeline_run: PipelineRun):
        """Setup and validate experimental environment"""
        self.logger.info("Stage 1: Environment setup")
        
        # Validate environment configuration
        env_config = self.load_environment_config()
        await self.validate_environment(env_config)
        
        # Setup containers and dependencies
        container_manager = ContainerManager(self.config)
        await container_manager.setup_experiment_environments()
        
        # Verify resource availability
        resources = await self.resource_scheduler.check_resource_availability()
        if not resources:
            raise RuntimeError("Insufficient compute resources available")
        
        pipeline_run.metrics["environment_setup_time"] = time.time()
    
    async def stage_run_tests(self, pipeline_run: PipelineRun):
        """Run unit and integration tests"""
        self.logger.info("Stage 2: Running tests")
        
        # Load test suite
        test_suite = self.load_test_suite()
        
        # Execute tests in parallel
        test_results = await self.test_runner.run_test_suite(test_suite)
        
        # Check for test failures
        failed_tests = [t for t in test_results if not t["passed"]]
        if failed_tests:
            error_msg = f"Tests failed: {[t['name'] for t in failed_tests]}"
            raise RuntimeError(error_msg)
        
        pipeline_run.tests = [t["test_id"] for t in test_results]
        pipeline_run.metrics["test_results"] = test_results
    
    async def stage_performance_tests(self, pipeline_run: PipelineRun):
        """Run performance regression tests"""
        self.logger.info("Stage 3: Performance regression tests")
        
        # Load performance baselines
        baselines = self.load_performance_baselines()
        
        # Run performance tests
        perf_results = await self.run_performance_tests(baselines)
        
        # Check for performance regressions
        regressions = self.detect_performance_regressions(perf_results, baselines)
        if regressions:
            self.logger.warning(f"Performance regressions detected: {regressions}")
            # Could fail pipeline or just warn based on config
        
        pipeline_run.metrics["performance_results"] = perf_results
    
    async def stage_run_experiments(self, pipeline_run: PipelineRun):
        """Run research experiments"""
        self.logger.info("Stage 4: Running experiments")
        
        # Load experiment configurations
        experiments = self.load_experiment_configs()
        
        # Execute experiments in parallel across available resources
        experiment_results = await self.experiment_manager.run_experiments_parallel(
            experiments, self.resource_scheduler
        )
        
        # Validate experiment outputs
        for result in experiment_results:
            if not result["success"]:
                self.logger.warning(f"Experiment {result['experiment_id']} failed")
        
        pipeline_run.experiments = [r["experiment_id"] for r in experiment_results]
        pipeline_run.metrics["experiment_results"] = experiment_results
    
    async def stage_validate_results(self, pipeline_run: PipelineRun):
        """Validate experimental results and generate artifacts"""
        self.logger.info("Stage 5: Validating results")
        
        # Validate experiment outputs
        validation_results = await self.validate_experiment_outputs(
            pipeline_run.metrics["experiment_results"]
        )
        
        # Generate artifacts
        artifacts = await self.artifact_manager.generate_artifacts(
            pipeline_run.experiments, validation_results
        )
        
        pipeline_run.artifacts = artifacts
        pipeline_run.metrics["validation_results"] = validation_results
    
    async def stage_performance_monitoring(self, pipeline_run: PipelineRun):
        """Monitor performance and trigger alerts if needed"""
        self.logger.info("Stage 6: Performance monitoring")
        
        # Collect performance metrics
        metrics = await self.performance_monitor.collect_metrics(pipeline_run)
        
        # Check for anomalies or significant changes
        anomalies = await self.performance_monitor.detect_anomalies(metrics)
        
        if anomalies:
            await self.send_performance_alerts(anomalies, pipeline_run)
        
        pipeline_run.metrics["monitoring_results"] = metrics
    
    async def stage_generate_reports(self, pipeline_run: PipelineRun):
        """Generate comprehensive reports"""
        self.logger.info("Stage 7: Generating reports")
        
        # Generate HTML report
        report_generator = ReportGenerator(self.config)
        html_report = await report_generator.generate_html_report(pipeline_run)
        
        # Generate performance dashboard
        dashboard = await report_generator.generate_dashboard(pipeline_run)
        
        # Save reports
        report_path = self.workspace_root / "reports" / f"run_{pipeline_run.run_id}"
        report_path.mkdir(exist_ok=True)
        
        with open(report_path / "report.html", "w") as f:
            f.write(html_report)
        
        with open(report_path / "dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
        
        pipeline_run.artifacts.extend([
            str(report_path / "report.html"),
            str(report_path / "dashboard.json")
        ])
    
    def generate_run_id(self, commit_hash: str, trigger_event: str) -> str:
        """Generate unique run ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_hash = commit_hash[:8]
        return f"{trigger_event}_{timestamp}_{short_hash}"
    
    def load_environment_config(self) -> Dict[str, Any]:
        """Load environment configuration"""
        config_path = self.workspace_root / "configs" / "environment.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return self.config.get("default_environment", {})
    
    def load_test_suite(self) -> List[TestCase]:
        """Load test suite configuration"""
        test_configs = []
        tests_dir = self.workspace_root / "tests"
        
        for test_file in tests_dir.glob("*.yaml"):
            with open(test_file) as f:
                test_data = yaml.safe_load(f)
                test_case = TestCase(**test_data)
                test_configs.append(test_case)
        
        return test_configs
    
    def load_experiment_configs(self) -> List[ExperimentConfig]:
        """Load experiment configurations"""
        experiment_configs = []
        experiments_dir = self.workspace_root / "experiments"
        
        for exp_file in experiments_dir.glob("*.yaml"):
            with open(exp_file) as f:
                exp_data = yaml.safe_load(f)
                exp_config = ExperimentConfig(**exp_data)
                experiment_configs.append(exp_config)
        
        return experiment_configs

class ExperimentManager:
    """Manages execution of research experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ExperimentManager")
        self.docker_client = docker.from_env()
        
    async def run_experiments_parallel(self, experiments: List[ExperimentConfig],
                                     resource_scheduler) -> List[Dict[str, Any]]:
        """Run multiple experiments in parallel"""
        
        # Schedule experiments across available resources
        scheduled_experiments = await resource_scheduler.schedule_experiments(experiments)
        
        # Execute experiments
        tasks = []
        for exp_config, resource in scheduled_experiments:
            task = asyncio.create_task(
                self.run_single_experiment(exp_config, resource)
            )
            tasks.append(task)
        
        # Wait for all experiments to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "experiment_id": experiments[i].experiment_id,
                    "success": False,
                    "error": str(result),
                    "outputs": []
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def run_single_experiment(self, experiment: ExperimentConfig,
                                  resource: ComputeResource) -> Dict[str, Any]:
        """Run a single experiment on specified resource"""
        
        self.logger.info(f"Running experiment {experiment.experiment_id} on {resource.resource_id}")
        
        try:
            # Prepare experiment environment
            container_config = self.prepare_experiment_container(experiment, resource)
            
            # Run experiment in container
            result = await self.execute_experiment_container(container_config)
            
            # Validate outputs
            outputs = await self.validate_experiment_outputs(experiment, result)
            
            return {
                "experiment_id": experiment.experiment_id,
                "success": True,
                "outputs": outputs,
                "resource_used": resource.resource_id,
                "execution_time": result.get("execution_time", 0),
                "metrics": result.get("metrics", {})
            }
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment.experiment_id} failed: {e}")
            return {
                "experiment_id": experiment.experiment_id,
                "success": False,
                "error": str(e),
                "outputs": []
            }
    
    def prepare_experiment_container(self, experiment: ExperimentConfig,
                                   resource: ComputeResource) -> Dict[str, Any]:
        """Prepare container configuration for experiment"""
        
        # Base container configuration
        container_config = {
            "image": experiment.environment.get("container_image", "hetero_sched:latest"),
            "command": f"python {experiment.script_path}",
            "environment": experiment.environment.get("env_vars", {}),
            "volumes": {
                str(Path.cwd()): {"bind": "/workspace", "mode": "rw"}
            },
            "working_dir": "/workspace"
        }
        
        # Add resource-specific configuration
        if resource.resource_type == ResourceType.GPU:
            container_config["runtime"] = "nvidia"
            container_config["environment"]["CUDA_VISIBLE_DEVICES"] = resource.specifications.get("gpu_ids", "0")
        
        # Add experiment parameters as environment variables
        for key, value in experiment.parameters.items():
            container_config["environment"][f"EXP_{key.upper()}"] = str(value)
        
        return container_config
    
    async def execute_experiment_container(self, container_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiment in Docker container"""
        
        start_time = datetime.datetime.now()
        
        # Run container
        container = self.docker_client.containers.run(
            detach=True,
            **container_config
        )
        
        # Wait for completion with timeout
        try:
            result = container.wait(timeout=7200)  # 2 hour timeout
            logs = container.logs().decode('utf-8')
            
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            return {
                "exit_code": result["StatusCode"],
                "logs": logs,
                "execution_time": execution_time,
                "success": result["StatusCode"] == 0
            }
            
        finally:
            container.remove(force=True)

class TestRunner:
    """Runs various types of tests in the CI pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("TestRunner")
    
    async def run_test_suite(self, test_suite: List[TestCase]) -> List[Dict[str, Any]]:
        """Run complete test suite"""
        
        # Group tests by dependencies
        test_groups = self.group_tests_by_dependencies(test_suite)
        
        results = []
        for group in test_groups:
            # Run tests in group in parallel
            group_tasks = [
                asyncio.create_task(self.run_single_test(test))
                for test in group
            ]
            
            group_results = await asyncio.gather(*group_tasks)
            results.extend(group_results)
        
        return results
    
    async def run_single_test(self, test: TestCase) -> Dict[str, Any]:
        """Run a single test case"""
        
        start_time = datetime.datetime.now()
        
        try:
            # Execute test script
            if test.test_type == TestType.UNIT:
                result = await self.run_unit_test(test)
            elif test.test_type == TestType.INTEGRATION:
                result = await self.run_integration_test(test)
            elif test.test_type == TestType.PERFORMANCE:
                result = await self.run_performance_test(test)
            elif test.test_type == TestType.REGRESSION:
                result = await self.run_regression_test(test)
            else:
                result = await self.run_generic_test(test)
            
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            return {
                "test_id": test.test_id,
                "name": test.name,
                "type": test.test_type.value,
                "passed": result["success"],
                "execution_time": execution_time,
                "outputs": result.get("outputs", []),
                "metrics": result.get("metrics", {})
            }
            
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            return {
                "test_id": test.test_id,
                "name": test.name,
                "type": test.test_type.value,
                "passed": False,
                "execution_time": execution_time,
                "error": str(e),
                "outputs": [],
                "metrics": {}
            }
    
    def group_tests_by_dependencies(self, tests: List[TestCase]) -> List[List[TestCase]]:
        """Group tests by dependency order"""
        # Simple implementation - could be enhanced with topological sort
        groups = []
        
        # Independent tests first
        independent = [t for t in tests if not t.dependencies]
        if independent:
            groups.append(independent)
        
        # Tests with dependencies
        dependent = [t for t in tests if t.dependencies]
        if dependent:
            groups.append(dependent)
        
        return groups

class ResourceScheduler:
    """Schedules experiments across available compute resources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ResourceScheduler")
        self.available_resources = self.initialize_resources()
    
    def initialize_resources(self) -> List[ComputeResource]:
        """Initialize available compute resources"""
        resources = []
        
        # Add CPU resources
        for i in range(self.config.get("num_cpu_workers", 2)):
            resource = ComputeResource(
                resource_id=f"cpu_worker_{i}",
                resource_type=ResourceType.CPU,
                specifications={"cores": 8, "memory_gb": 32},
                availability_schedule=[]
            )
            resources.append(resource)
        
        # Add GPU resources if available
        if self.config.get("enable_gpu", False):
            for i in range(self.config.get("num_gpu_workers", 1)):
                resource = ComputeResource(
                    resource_id=f"gpu_worker_{i}",
                    resource_type=ResourceType.GPU,
                    specifications={"gpu_type": "V100", "gpu_memory_gb": 16},
                    availability_schedule=[]
                )
                resources.append(resource)
        
        return resources
    
    async def schedule_experiments(self, experiments: List[ExperimentConfig]) -> List[Tuple[ExperimentConfig, ComputeResource]]:
        """Schedule experiments to available resources"""
        
        scheduled = []
        available = [r for r in self.available_resources if r.is_healthy]
        
        for experiment in experiments:
            # Find suitable resource
            resource = self.find_suitable_resource(experiment, available)
            if resource:
                scheduled.append((experiment, resource))
                # Mark resource as busy (simplified)
                resource.current_utilization = 1.0
            else:
                self.logger.warning(f"No suitable resource for experiment {experiment.experiment_id}")
        
        return scheduled
    
    def find_suitable_resource(self, experiment: ExperimentConfig, 
                             available: List[ComputeResource]) -> Optional[ComputeResource]:
        """Find suitable resource for experiment"""
        
        required_type = experiment.resource_requirements.get("type", "cpu")
        
        for resource in available:
            if resource.current_utilization < 0.8:  # Available capacity
                if required_type == "gpu" and resource.resource_type == ResourceType.GPU:
                    return resource
                elif required_type == "cpu" and resource.resource_type == ResourceType.CPU:
                    return resource
        
        return None

class ArtifactManager:
    """Manages experiment artifacts and results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ArtifactManager")
        
    async def generate_artifacts(self, experiment_ids: List[str], 
                               validation_results: Dict[str, Any]) -> List[str]:
        """Generate and store artifacts from experiments"""
        
        artifacts = []
        
        for exp_id in experiment_ids:
            # Collect experiment outputs
            exp_artifacts = await self.collect_experiment_artifacts(exp_id)
            artifacts.extend(exp_artifacts)
        
        # Generate summary artifacts
        summary_artifacts = await self.generate_summary_artifacts(validation_results)
        artifacts.extend(summary_artifacts)
        
        return artifacts
    
    async def collect_experiment_artifacts(self, experiment_id: str) -> List[str]:
        """Collect artifacts from a single experiment"""
        artifacts = []
        
        # Collect model checkpoints, logs, figures, etc.
        # This is simplified for demonstration
        
        return artifacts

def demonstrate_continuous_integration():
    """Demonstrate the continuous integration framework"""
    print("=== Continuous Integration for Research Experiments ===")
    
    # Configuration
    config = {
        "workspace_root": "ci_workspace",
        "num_cpu_workers": 2,
        "num_gpu_workers": 1,
        "enable_gpu": True,
        "container_registry": "localhost:5000",
        "notification_channels": ["email", "slack"],
        "performance_thresholds": {
            "training_time_max": 3600,
            "accuracy_min": 0.85,
            "memory_usage_max": 16
        }
    }
    
    print("1. Initializing CI Framework...")
    ci_framework = ContinuousIntegrationFramework(config)
    
    print("2. Creating Sample Experiment Configurations...")
    
    # Create sample experiment config
    experiment_config = ExperimentConfig(
        experiment_id="hetero_sched_baseline",
        name="HeteroSched Baseline Experiment",
        description="Baseline performance evaluation of HeteroSched system",
        script_path="experiments/baseline_evaluation.py",
        environment={
            "container_image": "hetero_sched:latest",
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
        },
        parameters={
            "num_episodes": 1000,
            "learning_rate": 0.001,
            "batch_size": 64,
            "num_environments": 4
        },
        resource_requirements={
            "type": "gpu",
            "memory_gb": 16,
            "gpu_memory_gb": 8
        },
        expected_outputs=[
            "model_checkpoint.pt",
            "training_metrics.json",
            "evaluation_results.json"
        ],
        performance_thresholds={
            "training_time": 1800,
            "final_reward": 0.8,
            "memory_usage": 12
        }
    )
    
    # Create sample test cases
    test_cases = [
        TestCase(
            test_id="unit_test_scheduler",
            name="Unit Test - Scheduler Components",
            test_type=TestType.UNIT,
            script_path="tests/test_scheduler.py",
            expected_outputs=["test_results.xml"],
            performance_metrics={"execution_time": 30}
        ),
        TestCase(
            test_id="integration_test_full_system",
            name="Integration Test - Full System",
            test_type=TestType.INTEGRATION,
            script_path="tests/test_integration.py",
            expected_outputs=["integration_results.json"],
            performance_metrics={"execution_time": 300}
        ),
        TestCase(
            test_id="performance_test_scheduling",
            name="Performance Test - Scheduling Latency",
            test_type=TestType.PERFORMANCE,
            script_path="tests/test_performance.py",
            expected_outputs=["performance_metrics.json"],
            performance_metrics={"max_latency_ms": 10, "throughput_ops_sec": 1000}
        )
    ]
    
    print("3. Demonstrating Resource Scheduling...")
    
    resource_scheduler = ResourceScheduler(config)
    print(f"   Available Resources: {len(resource_scheduler.available_resources)}")
    
    for resource in resource_scheduler.available_resources:
        print(f"     {resource.resource_id}: {resource.resource_type.value}")
        print(f"       Specs: {resource.specifications}")
        print(f"       Utilization: {resource.current_utilization:.1%}")
    
    print("4. Testing Experiment Manager...")
    
    experiment_manager = ExperimentManager(config)
    
    # Simulate experiment scheduling
    suitable_resource = resource_scheduler.find_suitable_resource(
        experiment_config, resource_scheduler.available_resources
    )
    
    if suitable_resource:
        print(f"   Experiment can be scheduled on: {suitable_resource.resource_id}")
    else:
        print("   No suitable resource found for experiment")
    
    print("5. Testing Test Runner...")
    
    test_runner = TestRunner(config)
    test_groups = test_runner.group_tests_by_dependencies(test_cases)
    
    print(f"   Test Groups: {len(test_groups)}")
    for i, group in enumerate(test_groups):
        print(f"     Group {i+1}: {[t.name for t in group]}")
    
    print("6. Demonstrating Pipeline Configuration...")
    
    # Sample pipeline configuration
    pipeline_config = {
        "trigger_events": ["push", "pull_request", "schedule"],
        "stages": [
            "environment_setup",
            "unit_tests", 
            "integration_tests",
            "performance_tests",
            "experiments",
            "validation",
            "monitoring",
            "reporting"
        ],
        "parallel_execution": True,
        "timeout_minutes": 240,
        "retry_policy": {
            "max_retries": 2,
            "retry_on": ["infrastructure_failure", "timeout"]
        },
        "notifications": {
            "on_success": ["email"],
            "on_failure": ["email", "slack"],
            "on_performance_regression": ["email", "slack", "pager"]
        }
    }
    
    print("   Pipeline Stages:")
    for stage in pipeline_config["stages"]:
        print(f"     - {stage}")
    
    print("7. Performance Monitoring Configuration...")
    
    monitoring_config = {
        "metrics": [
            "training_time",
            "model_accuracy", 
            "memory_usage",
            "gpu_utilization",
            "convergence_rate",
            "resource_efficiency"
        ],
        "thresholds": {
            "training_time_increase": 0.2,  # 20% increase
            "accuracy_decrease": 0.05,      # 5% decrease
            "memory_increase": 0.3          # 30% increase
        },
        "anomaly_detection": {
            "enabled": True,
            "sensitivity": "medium",
            "baseline_window_days": 7
        }
    }
    
    print("   Monitored Metrics:")
    for metric in monitoring_config["metrics"]:
        print(f"     - {metric}")
    
    print("8. Artifact Management...")
    
    artifact_manager = ArtifactManager(config)
    
    artifact_types = [
        "Model checkpoints with versioning",
        "Training logs and metrics",
        "Evaluation results and plots", 
        "System performance profiles",
        "Configuration snapshots",
        "Test coverage reports",
        "Performance benchmark results"
    ]
    
    print("   Managed Artifact Types:")
    for artifact_type in artifact_types:
        print(f"     - {artifact_type}")
    
    print("\n=== CI/CD Benefits for Research ===")
    
    benefits = [
        "Automated validation of research experiments across hardware",
        "Reproducible experiments with containerized environments",
        "Performance regression detection for model improvements",
        "Parallel execution across heterogeneous compute resources",
        "Comprehensive artifact management and versioning",
        "Integration with git workflow for research collaboration",
        "Automated alerting for performance anomalies",
        "Standardized reporting and result visualization"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit}")
    
    print("\n=== Integration Points ===")
    
    integration_points = [
        "Git hooks for automatic pipeline triggering",
        "Docker containers for reproducible environments", 
        "Kubernetes for distributed experiment execution",
        "MLflow for experiment tracking and model registry",
        "Prometheus/Grafana for performance monitoring",
        "Slack/Email for notification delivery",
        "S3/MinIO for artifact storage",
        "SLURM/Kubernetes for resource management"
    ]
    
    for i, point in enumerate(integration_points, 1):
        print(f"{i}. {point}")
    
    return {
        "ci_framework": ci_framework,
        "experiment_config": experiment_config,
        "test_cases": test_cases,
        "resource_scheduler": resource_scheduler,
        "pipeline_config": pipeline_config
    }

if __name__ == "__main__":
    demonstrate_continuous_integration()