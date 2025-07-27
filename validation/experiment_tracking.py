"""
Experiment Tracking and Artifact Management System

This module implements R38: comprehensive experiment tracking and artifact 
management system that enables reproducible research and systematic 
experiment organization for the HeteroSched project.

Key Features:
1. Comprehensive experiment metadata tracking
2. Artifact versioning and storage management
3. Experiment provenance and lineage tracking
4. Automated model and data artifact collection
5. Experiment comparison and analysis tools
6. Integration with version control systems
7. Distributed experiment coordination
8. Results aggregation and reporting

The system ensures reproducibility and enables systematic analysis
of experimental results across the research pipeline.

Authors: HeteroSched Research Team
"""

import os
import json
import pickle
import hashlib
import logging
import time
import datetime
import sqlite3
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import git
from contextlib import contextmanager

class ExperimentStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ArtifactType(Enum):
    MODEL = "model"
    DATA = "data"
    CONFIG = "config"
    RESULTS = "results"
    LOGS = "logs"
    PLOTS = "plots"
    CODE = "code"
    METRICS = "metrics"

class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_name: str
    experiment_type: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class Artifact:
    """Represents an experiment artifact"""
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    file_path: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
@dataclass
class ExperimentRun:
    """Represents a single experiment run"""
    run_id: str
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    start_time: float
    end_time: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    git_commit: Optional[str] = None
    parent_run_id: Optional[str] = None
    child_run_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

@dataclass
class ExperimentSeries:
    """Represents a series of related experiments"""
    series_id: str
    name: str
    description: str
    run_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArtifactStorage:
    """Handles artifact storage and retrieval"""
    
    def __init__(self, storage_backend: StorageBackend, base_path: str):
        self.storage_backend = storage_backend
        self.base_path = Path(base_path)
        self.logger = logging.getLogger("ArtifactStorage")
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def store_artifact(self, artifact: Artifact, source_path: str) -> str:
        """Store an artifact and return the storage path"""
        
        # Create directory structure
        artifact_dir = self.base_path / artifact.artifact_type.value / artifact.artifact_id[:2]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine storage path
        storage_path = artifact_dir / f"{artifact.artifact_id}_{artifact.name}"
        
        if self.storage_backend == StorageBackend.LOCAL:
            return self._store_local(source_path, storage_path)
        else:
            raise NotImplementedError(f"Storage backend {self.storage_backend} not implemented")
    
    def _store_local(self, source_path: str, storage_path: Path) -> str:
        """Store artifact locally"""
        
        source = Path(source_path)
        
        try:
            if source.is_file():
                shutil.copy2(source, storage_path)
            elif source.is_dir():
                # Create zip archive for directories
                storage_path = storage_path.with_suffix('.zip')
                with zipfile.ZipFile(storage_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in source.rglob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.relative_to(source))
            else:
                raise ValueError(f"Source path {source_path} does not exist")
            
            self.logger.info(f"Stored artifact at {storage_path}")
            return str(storage_path)
            
        except Exception as e:
            self.logger.error(f"Failed to store artifact: {e}")
            raise
    
    def retrieve_artifact(self, artifact: Artifact, destination_path: str) -> bool:
        """Retrieve an artifact to destination path"""
        
        storage_path = Path(artifact.file_path)
        destination = Path(destination_path)
        
        try:
            if storage_path.suffix == '.zip':
                # Extract zip archive
                with zipfile.ZipFile(storage_path, 'r') as zipf:
                    zipf.extractall(destination)
            else:
                shutil.copy2(storage_path, destination)
            
            self.logger.info(f"Retrieved artifact from {storage_path} to {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact: {e}")
            return False
    
    def delete_artifact(self, artifact: Artifact) -> bool:
        """Delete an artifact"""
        
        storage_path = Path(artifact.file_path)
        
        try:
            if storage_path.exists():
                storage_path.unlink()
                self.logger.info(f"Deleted artifact {storage_path}")
                return True
            else:
                self.logger.warning(f"Artifact {storage_path} does not exist")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete artifact: {e}")
            return False

class ExperimentDatabase:
    """SQLite database for experiment metadata"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger("ExperimentDatabase")
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    created_at REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    metrics TEXT,
                    git_commit TEXT,
                    parent_run_id TEXT,
                    error_message TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_series (
                    series_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    run_ids TEXT,
                    created_at REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (experiment_id, name, experiment_type, description, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                config.experiment_name,
                config.experiment_type,
                config.description,
                json.dumps(config.tags),
                time.time()
            ))
            conn.commit()
        
        self.logger.info(f"Created experiment {experiment_id}: {config.experiment_name}")
        return experiment_id
    
    def create_run(self, experiment_id: str, config: ExperimentConfig, 
                   git_commit: Optional[str] = None, parent_run_id: Optional[str] = None) -> str:
        """Create a new experiment run"""
        
        run_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiment_runs (
                    run_id, experiment_id, config, status, start_time, 
                    git_commit, parent_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                experiment_id,
                json.dumps(asdict(config)),
                ExperimentStatus.CREATED.value,
                time.time(),
                git_commit,
                parent_run_id
            ))
            conn.commit()
        
        self.logger.info(f"Created run {run_id} for experiment {experiment_id}")
        return run_id
    
    def update_run_status(self, run_id: str, status: ExperimentStatus, 
                         end_time: Optional[float] = None, error_message: Optional[str] = None):
        """Update run status"""
        
        with sqlite3.connect(self.db_path) as conn:
            if end_time is not None:
                conn.execute("""
                    UPDATE experiment_runs 
                    SET status = ?, end_time = ?, error_message = ?
                    WHERE run_id = ?
                """, (status.value, end_time, error_message, run_id))
            else:
                conn.execute("""
                    UPDATE experiment_runs 
                    SET status = ?, error_message = ?
                    WHERE run_id = ?
                """, (status.value, error_message, run_id))
            conn.commit()
    
    def update_run_metrics(self, run_id: str, metrics: Dict[str, Any]):
        """Update run metrics"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE experiment_runs 
                SET metrics = ?
                WHERE run_id = ?
            """, (json.dumps(metrics), run_id))
            conn.commit()
    
    def add_artifact(self, run_id: str, artifact: Artifact):
        """Add artifact to run"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO artifacts (
                    artifact_id, run_id, name, artifact_type, file_path,
                    size_bytes, checksum, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact.artifact_id,
                run_id,
                artifact.name,
                artifact.artifact_type.value,
                artifact.file_path,
                artifact.size_bytes,
                artifact.checksum,
                json.dumps(artifact.metadata),
                artifact.created_at
            ))
            conn.commit()
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT run_id, experiment_id, config, status, start_time, end_time,
                       metrics, git_commit, parent_run_id, error_message
                FROM experiment_runs
                WHERE run_id = ?
            """, (run_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get artifacts
            artifacts_cursor = conn.execute("""
                SELECT artifact_id, name, artifact_type, file_path, size_bytes,
                       checksum, metadata, created_at
                FROM artifacts
                WHERE run_id = ?
            """, (run_id,))
            
            artifacts = []
            for artifact_row in artifacts_cursor.fetchall():
                artifact = Artifact(
                    artifact_id=artifact_row[0],
                    name=artifact_row[1],
                    artifact_type=ArtifactType(artifact_row[2]),
                    file_path=artifact_row[3],
                    size_bytes=artifact_row[4],
                    checksum=artifact_row[5],
                    metadata=json.loads(artifact_row[6]) if artifact_row[6] else {},
                    created_at=artifact_row[7]
                )
                artifacts.append(artifact)
            
            # Construct ExperimentRun
            config_dict = json.loads(row[2])
            config = ExperimentConfig(**config_dict)
            
            experiment_run = ExperimentRun(
                run_id=row[0],
                experiment_id=row[1],
                config=config,
                status=ExperimentStatus(row[3]),
                start_time=row[4],
                end_time=row[5],
                metrics=json.loads(row[6]) if row[6] else {},
                artifacts=artifacts,
                git_commit=row[7],
                parent_run_id=row[8],
                error_message=row[9]
            )
            
            return experiment_run
    
    def list_runs(self, experiment_id: Optional[str] = None, 
                  status: Optional[ExperimentStatus] = None) -> List[str]:
        """List experiment runs"""
        
        query = "SELECT run_id FROM experiment_runs"
        params = []
        conditions = []
        
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY start_time DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]
    
    def create_series(self, series: ExperimentSeries) -> str:
        """Create experiment series"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiment_series (
                    series_id, name, description, run_ids, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                series.series_id,
                series.name,
                series.description,
                json.dumps(series.run_ids),
                series.created_at,
                json.dumps(series.metadata)
            ))
            conn.commit()
        
        return series.series_id

class ExperimentTracker:
    """Main experiment tracking system"""
    
    def __init__(self, base_path: str, storage_backend: StorageBackend = StorageBackend.LOCAL):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("ExperimentTracker")
        
        # Initialize components
        self.database = ExperimentDatabase(str(self.base_path / "experiments.db"))
        self.storage = ArtifactStorage(storage_backend, str(self.base_path / "artifacts"))
        
        # Active run context
        self.current_run: Optional[ExperimentRun] = None
        self._run_lock = threading.Lock()
        
        # Auto-logging configuration
        self.auto_log_artifacts = True
        self.auto_log_metrics = True
        
    @contextmanager
    def start_run(self, config: ExperimentConfig, experiment_id: Optional[str] = None):
        """Start an experiment run context"""
        
        # Create experiment if needed
        if experiment_id is None:
            experiment_id = self.database.create_experiment(config)
        
        # Get git commit if available
        git_commit = self._get_git_commit()
        
        # Create run
        run_id = self.database.create_run(experiment_id, config, git_commit)
        
        # Create run object
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            start_time=time.time(),
            git_commit=git_commit
        )
        
        with self._run_lock:
            self.current_run = run
        
        # Update status
        self.database.update_run_status(run_id, ExperimentStatus.RUNNING)
        
        self.logger.info(f"Started run {run_id}")
        
        try:
            yield run
            
            # Mark as completed
            self.database.update_run_status(run_id, ExperimentStatus.COMPLETED, time.time())
            run.status = ExperimentStatus.COMPLETED
            run.end_time = time.time()
            
            self.logger.info(f"Completed run {run_id}")
            
        except Exception as e:
            # Mark as failed
            error_message = str(e)
            self.database.update_run_status(run_id, ExperimentStatus.FAILED, time.time(), error_message)
            run.status = ExperimentStatus.FAILED
            run.end_time = time.time()
            run.error_message = error_message
            
            self.logger.error(f"Failed run {run_id}: {error_message}")
            raise
            
        finally:
            with self._run_lock:
                self.current_run = None
    
    def log_metric(self, name: str, value: Union[int, float], step: Optional[int] = None):
        """Log a metric value"""
        
        if self.current_run is None:
            self.logger.warning("No active run for metric logging")
            return
        
        if step is not None:
            metric_key = f"{name}_step_{step}"
        else:
            metric_key = name
        
        self.current_run.metrics[metric_key] = value
        
        # Update database
        self.database.update_run_metrics(self.current_run.run_id, self.current_run.metrics)
        
        self.logger.debug(f"Logged metric {name}={value}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Log multiple metrics"""
        
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_artifact(self, name: str, file_path: str, artifact_type: ArtifactType,
                    metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Log an artifact"""
        
        if self.current_run is None:
            raise ValueError("No active run for artifact logging")
        
        # Calculate file info
        source_path = Path(file_path)
        if not source_path.exists():
            raise ValueError(f"Artifact file {file_path} does not exist")
        
        size_bytes = self._get_size(source_path)
        checksum = self._calculate_checksum(source_path)
        
        # Create artifact
        artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            name=name,
            artifact_type=artifact_type,
            file_path="",  # Will be set by storage
            size_bytes=size_bytes,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        # Store artifact
        storage_path = self.storage.store_artifact(artifact, file_path)
        artifact.file_path = storage_path
        
        # Add to run and database
        self.current_run.artifacts.append(artifact)
        self.database.add_artifact(self.current_run.run_id, artifact)
        
        self.logger.info(f"Logged artifact {name} ({artifact_type.value})")
        
        return artifact
    
    def log_model(self, model_path: str, name: str = "model",
                 metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Log a model artifact"""
        return self.log_artifact(name, model_path, ArtifactType.MODEL, metadata)
    
    def log_config(self, config_path: str, name: str = "config",
                  metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Log a config artifact"""
        return self.log_artifact(name, config_path, ArtifactType.CONFIG, metadata)
    
    def log_results(self, results_path: str, name: str = "results",
                   metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Log results artifact"""
        return self.log_artifact(name, results_path, ArtifactType.RESULTS, metadata)
    
    def log_plot(self, plot_path: str, name: str = "plot",
                metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Log a plot artifact"""
        return self.log_artifact(name, plot_path, ArtifactType.PLOTS, metadata)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.commit.hexsha
        except:
            return None
    
    def _get_size(self, path: Path) -> int:
        """Get file or directory size"""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        else:
            return 0
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum"""
        
        hash_md5 = hashlib.md5()
        
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        elif path.is_dir():
            # Hash directory structure and file contents
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    relative_path = file_path.relative_to(path)
                    hash_md5.update(str(relative_path).encode())
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run"""
        return self.database.get_run(run_id)
    
    def list_runs(self, experiment_id: Optional[str] = None, 
                  status: Optional[ExperimentStatus] = None) -> List[str]:
        """List experiment runs"""
        return self.database.list_runs(experiment_id, status)
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiment runs"""
        
        runs_data = []
        
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            
            run_data = {
                "run_id": run_id,
                "experiment_id": run.experiment_id,
                "status": run.status.value,
                "start_time": datetime.datetime.fromtimestamp(run.start_time),
                "duration": (run.end_time - run.start_time) if run.end_time else None,
                "git_commit": run.git_commit
            }
            
            # Add metrics
            for metric_name, metric_value in run.metrics.items():
                run_data[f"metric_{metric_name}"] = metric_value
            
            # Add config parameters
            for param_name, param_value in run.config.parameters.items():
                run_data[f"param_{param_name}"] = param_value
            
            runs_data.append(run_data)
        
        return pd.DataFrame(runs_data)
    
    def create_series(self, name: str, run_ids: List[str], 
                     description: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create experiment series"""
        
        series = ExperimentSeries(
            series_id=str(uuid.uuid4()),
            name=name,
            description=description,
            run_ids=run_ids,
            metadata=metadata or {}
        )
        
        return self.database.create_series(series)

class ExperimentAnalyzer:
    """Analyzes experiment results and generates reports"""
    
    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.logger = logging.getLogger("ExperimentAnalyzer")
    
    def generate_summary_report(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary report for experiments"""
        
        run_ids = self.tracker.list_runs(experiment_id)
        
        if not run_ids:
            return {"message": "No runs found"}
        
        runs_df = self.tracker.compare_runs(run_ids)
        
        # Basic statistics
        total_runs = len(runs_df)
        completed_runs = len(runs_df[runs_df['status'] == 'completed'])
        failed_runs = len(runs_df[runs_df['status'] == 'failed'])
        
        # Duration statistics
        durations = runs_df['duration'].dropna()
        
        # Convert timedelta to seconds for averaging
        duration_seconds = []
        for duration in durations:
            if hasattr(duration, 'total_seconds'):
                duration_seconds.append(duration.total_seconds())
            else:
                duration_seconds.append(float(duration))
        
        avg_duration_seconds = np.mean(duration_seconds) if duration_seconds else None
        
        # Metric statistics
        metric_columns = [col for col in runs_df.columns if col.startswith('metric_')]
        metric_stats = {}
        
        for col in metric_columns:
            metric_name = col.replace('metric_', '')
            values = runs_df[col].dropna()
            if len(values) > 0:
                metric_stats[metric_name] = {
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max())
                }
        
        report = {
            "experiment_id": experiment_id,
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0,
            "average_duration_seconds": avg_duration_seconds,
            "metric_statistics": metric_stats,
            "runs_analyzed": run_ids
        }
        
        return report
    
    def plot_metric_trends(self, metric_name: str, run_ids: Optional[List[str]] = None, 
                          save_path: Optional[str] = None) -> str:
        """Plot metric trends across runs"""
        
        if run_ids is None:
            run_ids = self.tracker.list_runs()
        
        runs_df = self.tracker.compare_runs(run_ids)
        
        metric_col = f"metric_{metric_name}"
        if metric_col not in runs_df.columns:
            raise ValueError(f"Metric {metric_name} not found in runs")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot metric values over time
        plt.subplot(1, 2, 1)
        plt.plot(runs_df['start_time'], runs_df[metric_col], 'o-')
        plt.title(f'{metric_name} Over Time')
        plt.xlabel('Run Start Time')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        
        # Plot metric distribution
        plt.subplot(1, 2, 2)
        runs_df[metric_col].hist(bins=20)
        plt.title(f'{metric_name} Distribution')
        plt.xlabel(metric_name)
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved metric plot to {save_path}")
            return save_path
        else:
            # Save to temporary file
            import tempfile
            temp_path = tempfile.mktemp(suffix='.png')
            plt.savefig(temp_path)
            plt.close()
            return temp_path
    
    def find_best_runs(self, metric_name: str, top_k: int = 5, 
                      minimize: bool = False) -> List[Tuple[str, float]]:
        """Find best performing runs based on a metric"""
        
        run_ids = self.tracker.list_runs()
        runs_df = self.tracker.compare_runs(run_ids)
        
        metric_col = f"metric_{metric_name}"
        if metric_col not in runs_df.columns:
            return []
        
        # Sort runs by metric
        sorted_runs = runs_df.sort_values(metric_col, ascending=minimize)
        
        # Get top k runs
        top_runs = sorted_runs.head(top_k)
        
        return [(row['run_id'], row[metric_col]) for _, row in top_runs.iterrows()]

def demonstrate_experiment_tracking():
    """Demonstrate the experiment tracking system"""
    print("=== Experiment Tracking and Artifact Management System ===")
    
    # Setup
    base_path = "D:/temp/hetero_sched_experiments"
    
    print("1. Initializing Experiment Tracking System...")
    
    tracker = ExperimentTracker(base_path)
    analyzer = ExperimentAnalyzer(tracker)
    
    print(f"   Base path: {base_path}")
    print("   Database and storage initialized")
    
    print("2. Running Sample Experiments...")
    
    # Experiment 1: Baseline scheduling
    config1 = ExperimentConfig(
        experiment_name="baseline_scheduling",
        experiment_type="scheduling_policy",
        description="Baseline scheduling policy evaluation",
        tags=["baseline", "scheduling"],
        parameters={
            "learning_rate": 0.001,
            "batch_size": 64,
            "hidden_dim": 256,
            "scheduler_type": "fifo"
        }
    )
    
    with tracker.start_run(config1) as run1:
        print(f"   Started run {run1.run_id[:8]}... for baseline experiment")
        
        # Simulate training metrics
        for epoch in range(5):
            makespan = 100 - epoch * 5 + np.random.normal(0, 2)
            utilization = 0.6 + epoch * 0.05 + np.random.normal(0, 0.02)
            
            tracker.log_metric("makespan", makespan, epoch)
            tracker.log_metric("utilization", utilization, epoch)
        
        # Log final metrics
        tracker.log_metrics({
            "final_makespan": 85.2,
            "final_utilization": 0.82,
            "training_time": 300.5
        })
        
        # Create and log artifacts
        import tempfile
        
        # Model artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"model_type": "baseline", "parameters": config1.parameters}, f)
            model_path = f.name
        
        tracker.log_model(model_path, "baseline_model", 
                         metadata={"model_type": "fifo_scheduler"})
        
        # Results artifact
        results = {
            "final_metrics": {"makespan": 85.2, "utilization": 0.82},
            "convergence_epoch": 4,
            "best_performance": {"makespan": 82.1, "utilization": 0.84}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(results, f)
            results_path = f.name
        
        tracker.log_results(results_path, "experiment_results")
        
        print(f"     Logged {len(run1.metrics)} metrics and {len(run1.artifacts)} artifacts")
    
    # Experiment 2: Deep RL scheduling
    config2 = ExperimentConfig(
        experiment_name="deep_rl_scheduling",
        experiment_type="scheduling_policy",
        description="Deep RL based scheduling policy",
        tags=["deep_rl", "scheduling", "experimental"],
        parameters={
            "learning_rate": 0.0005,
            "batch_size": 128,
            "hidden_dim": 512,
            "scheduler_type": "deep_rl",
            "reward_function": "multi_objective"
        }
    )
    
    with tracker.start_run(config2) as run2:
        print(f"   Started run {run2.run_id[:8]}... for deep RL experiment")
        
        # Simulate training metrics with improvement
        for epoch in range(5):
            makespan = 90 - epoch * 8 + np.random.normal(0, 1.5)
            utilization = 0.65 + epoch * 0.08 + np.random.normal(0, 0.015)
            
            tracker.log_metric("makespan", makespan, epoch)
            tracker.log_metric("utilization", utilization, epoch)
        
        # Log final metrics
        tracker.log_metrics({
            "final_makespan": 58.3,
            "final_utilization": 0.94,
            "training_time": 1200.8
        })
        
        # Create model artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"model_type": "deep_rl", "parameters": config2.parameters}, f)
            model_path = f.name
        
        tracker.log_model(model_path, "deep_rl_model", 
                         metadata={"model_type": "multi_objective_dqn"})
        
        print(f"     Logged {len(run2.metrics)} metrics and {len(run2.artifacts)} artifacts")
    
    # Experiment 3: Failed experiment (for demonstration)
    config3 = ExperimentConfig(
        experiment_name="experimental_scheduling",
        experiment_type="scheduling_policy",
        description="Experimental scheduling approach",
        tags=["experimental"],
        parameters={"learning_rate": 0.1, "experimental_param": True}
    )
    
    try:
        with tracker.start_run(config3) as run3:
            print(f"   Started run {run3.run_id[:8]}... for experimental approach")
            
            # Simulate some metrics before failure
            tracker.log_metric("makespan", 95.0, 0)
            tracker.log_metric("utilization", 0.7, 0)
            
            # Simulate failure
            raise RuntimeError("Simulated experimental failure")
            
    except RuntimeError:
        print(f"     Run failed as expected (simulated)")
    
    print("3. Analyzing Experiment Results...")
    
    # List all runs
    all_runs = tracker.list_runs()
    print(f"   Total runs: {len(all_runs)}")
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    print(f"   Completed runs: {summary['completed_runs']}")
    print(f"   Failed runs: {summary['failed_runs']}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    
    if summary['average_duration_seconds']:
        print(f"   Average duration: {summary['average_duration_seconds']:.1f}s")
    
    print("4. Metric Analysis...")
    
    if 'final_makespan' in summary['metric_statistics']:
        makespan_stats = summary['metric_statistics']['final_makespan']
        print(f"   Final makespan - Mean: {makespan_stats['mean']:.1f}, "
              f"Best: {makespan_stats['min']:.1f}")
    
    if 'final_utilization' in summary['metric_statistics']:
        util_stats = summary['metric_statistics']['final_utilization']
        print(f"   Final utilization - Mean: {util_stats['mean']:.3f}, "
              f"Best: {util_stats['max']:.3f}")
    
    print("5. Finding Best Performing Runs...")
    
    # Find best runs by makespan (minimize)
    best_makespan_runs = analyzer.find_best_runs("final_makespan", top_k=3, minimize=True)
    print("   Best makespan runs:")
    for i, (run_id, value) in enumerate(best_makespan_runs, 1):
        print(f"     {i}. Run {run_id[:8]}...: {value:.1f}")
    
    # Find best runs by utilization (maximize)
    best_util_runs = analyzer.find_best_runs("final_utilization", top_k=3, minimize=False)
    print("   Best utilization runs:")
    for i, (run_id, value) in enumerate(best_util_runs, 1):
        print(f"     {i}. Run {run_id[:8]}...: {value:.3f}")
    
    print("6. Comparing Runs...")
    
    comparison_df = tracker.compare_runs(all_runs[:2])  # Compare first 2 successful runs
    print("   Run comparison (first 2 successful runs):")
    
    if len(comparison_df) >= 2:
        for _, row in comparison_df.iterrows():
            print(f"     Run {row['run_id'][:8]}...: "
                  f"makespan={row.get('metric_final_makespan', 'N/A')}, "
                  f"utilization={row.get('metric_final_utilization', 'N/A')}")
    
    print("7. Artifact Management...")
    
    # Show artifacts from runs
    for run_id in all_runs[:2]:  # First 2 runs
        run = tracker.get_run(run_id)
        if run and run.artifacts:
            print(f"   Run {run_id[:8]}... artifacts:")
            for artifact in run.artifacts:
                size_mb = artifact.size_bytes / (1024 * 1024)
                print(f"     {artifact.name} ({artifact.artifact_type.value}): "
                      f"{size_mb:.2f} MB")
    
    print("8. Experiment Series Creation...")
    
    # Create experiment series
    scheduling_runs = [run_id for run_id in all_runs 
                      if tracker.get_run(run_id) and 
                      tracker.get_run(run_id).config.experiment_type == "scheduling_policy"]
    
    if len(scheduling_runs) >= 2:
        series_id = tracker.create_series(
            "Scheduling Policy Comparison",
            scheduling_runs[:2],
            "Comparison of different scheduling policies",
            metadata={"focus": "policy_comparison", "date": "2024"}
        )
        print(f"   Created experiment series: {series_id[:8]}...")
    
    print("9. System Statistics...")
    
    # Database statistics
    total_experiments = len(set(tracker.get_run(run_id).experiment_id 
                               for run_id in all_runs 
                               if tracker.get_run(run_id)))
    
    total_artifacts = sum(len(tracker.get_run(run_id).artifacts) 
                         for run_id in all_runs 
                         if tracker.get_run(run_id))
    
    print(f"   Total experiments: {total_experiments}")
    print(f"   Total runs: {len(all_runs)}")
    print(f"   Total artifacts: {total_artifacts}")
    
    print("10. Framework Benefits...")
    
    benefits = [
        "Comprehensive experiment metadata and provenance tracking",
        "Automated artifact collection and versioning",
        "Git integration for code version tracking",
        "Statistical analysis and comparison tools",
        "Reproducible experiment environments",
        "Scalable storage backend support",
        "Real-time metric logging and visualization",
        "Experiment series organization and management"
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"   {i}. {benefit}")
    
    # Cleanup temporary files
    try:
        os.unlink(model_path)
        os.unlink(results_path)
    except:
        pass
    
    return {
        "tracker": tracker,
        "analyzer": analyzer,
        "summary": summary,
        "all_runs": all_runs,
        "comparison_df": comparison_df
    }

if __name__ == "__main__":
    demonstrate_experiment_tracking()