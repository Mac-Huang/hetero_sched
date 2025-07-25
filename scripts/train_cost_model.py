#!/usr/bin/env python3
"""
HeteroSched ML Cost Model Training

This script trains three machine learning models for the heterogeneous scheduler:
1. model_cpu.pkl: Linear regression predicting CPU runtime
2. model_gpu.pkl: Linear regression predicting GPU runtime  
3. model_route.pkl: Binary classifier for CPU vs GPU routing

Author: MLSys Research
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

class HeteroSchedMLTrainer:
    def __init__(self, csv_path='logs/task_log.csv', models_dir='models'):
        self.csv_path = csv_path
        self.models_dir = models_dir
        self.task_type_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load CSV data and perform feature engineering"""
        print(f"Loading data from {self.csv_path}...")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: CSV file {self.csv_path} not found!")
            print("Please run the hetero_sched binary first to generate training data.")
            sys.exit(1)
            
        # Load CSV data
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} task records")
        
        # Basic data validation
        if len(df) < 10:
            print("Warning: Very little training data available. Run more tasks for better models.")
            
        # Display data overview
        print("\nData Overview:")
        print(df.head())
        print(f"\nTask type distribution:")
        print(df['task_type'].value_counts())
        print(f"Device distribution:")
        print(df['device'].value_counts())
        
        # Feature engineering
        print("\nPerforming feature engineering...")
        
        # One-hot encode task types
        task_type_dummies = pd.get_dummies(df['task_type'], prefix='task')
        
        # Normalize task size (log scale for better ML performance)
        df['log_task_size'] = np.log1p(df['task_size'])
        df['log_rows'] = np.log1p(df['rows']) 
        df['log_cols'] = np.log1p(df['cols'])
        
        # Compute task complexity metrics
        df['complexity_score'] = df['task_size'] * np.where(df['task_type'] == 'MATMUL', df['rows'], 1)
        df['log_complexity'] = np.log1p(df['complexity_score'])
        
        # Create feature matrix
        feature_columns = ['log_task_size', 'log_rows', 'log_cols', 'log_complexity']
        X = df[feature_columns].copy()
        X = pd.concat([X, task_type_dummies], axis=1)
        
        # Target variables
        y_time = df['execution_time_ms']
        y_device = df['device']
        
        # Create separate datasets for CPU and GPU timing models
        cpu_mask = df['device'] == 'CPU'
        gpu_mask = df['device'] == 'GPU'
        
        X_cpu = X[cpu_mask]
        y_cpu_time = y_time[cpu_mask] 
        
        X_gpu = X[gpu_mask]
        y_gpu_time = y_time[gpu_mask]
        
        print(f"CPU training samples: {len(X_cpu)}")
        print(f"GPU training samples: {len(X_gpu)}")
        
        return X, y_device, X_cpu, y_cpu_time, X_gpu, y_gpu_time, df
        
    def train_cpu_model(self, X_cpu, y_cpu_time):
        """Train CPU runtime prediction model"""
        print("\n=== Training CPU Runtime Model ===")
        
        if len(X_cpu) < 5:
            print("Warning: Insufficient CPU training data")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_cpu, y_cpu_time, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"CPU Model RMSE: {rmse:.3f} ms")
        print(f"Mean CPU time: {y_cpu_time.mean():.3f} ms")
        print(f"Relative error: {rmse/y_cpu_time.mean()*100:.1f}%")
        
        # Save model and scaler
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X_cpu.columns),
            'rmse': rmse
        }
        
        with open(f'{self.models_dir}/model_cpu.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data
        
    def train_gpu_model(self, X_gpu, y_gpu_time):
        """Train GPU runtime prediction model"""
        print("\n=== Training GPU Runtime Model ===")
        
        if len(X_gpu) < 5:
            print("Warning: Insufficient GPU training data")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_gpu, y_gpu_time, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"GPU Model RMSE: {rmse:.3f} ms")
        print(f"Mean GPU time: {y_gpu_time.mean():.3f} ms")
        print(f"Relative error: {rmse/y_gpu_time.mean()*100:.1f}%")
        
        # Save model and scaler
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X_gpu.columns),
            'rmse': rmse
        }
        
        with open(f'{self.models_dir}/model_gpu.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data
        
    def train_routing_model(self, X, y_device):
        """Train CPU vs GPU routing classifier"""
        print("\n=== Training Device Routing Model ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_device, test_size=0.2, random_state=42, stratify=y_device
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple classifiers
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_accuracy = 0
        best_name = None
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n{name} Accuracy: {accuracy:.3f}")
            print(f"Classification Report:")
            print(classification_report(y_test, y_pred))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_name = name
                
        print(f"\nBest model: {best_name} (Accuracy: {best_accuracy:.3f})")
        
        # Save best model and scaler
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'accuracy': best_accuracy,
            'model_type': best_name
        }
        
        with open(f'{self.models_dir}/model_route.pkl', 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data
        
    def generate_plots(self, df):
        """Generate visualization plots"""
        print("\n=== Generating Visualization Plots ===")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Task size vs execution time by device
        ax1 = axes[0, 0]
        for device in df['device'].unique():
            device_data = df[df['device'] == device]
            ax1.scatter(device_data['task_size'], device_data['execution_time_ms'], 
                       label=device, alpha=0.6)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Task Size')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Task Size vs Execution Time')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Task type distribution
        ax2 = axes[0, 1]
        task_counts = df['task_type'].value_counts()
        ax2.bar(task_counts.index, task_counts.values)
        ax2.set_xlabel('Task Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Task Type Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Device utilization
        ax3 = axes[1, 0]
        device_counts = df['device'].value_counts()
        ax3.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
        ax3.set_title('Device Utilization')
        
        # Plot 4: Performance comparison
        ax4 = axes[1, 1]
        cpu_times = df[df['device'] == 'CPU']['execution_time_ms']
        gpu_times = df[df['device'] == 'GPU']['execution_time_ms']
        
        ax4.hist(cpu_times, bins=20, alpha=0.7, label='CPU', density=True)
        ax4.hist(gpu_times, bins=20, alpha=0.7, label='GPU', density=True)
        ax4.set_xlabel('Execution Time (ms)')
        ax4.set_ylabel('Density')
        ax4.set_title('Execution Time Distribution')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.models_dir}/training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to {self.models_dir}/training_analysis.png")
        
    def train_all_models(self):
        """Main training pipeline"""
        print("=== HeteroSched ML Model Training ===")
        
        # Load and preprocess data
        X, y_device, X_cpu, y_cpu_time, X_gpu, y_gpu_time, df = self.load_and_preprocess_data()
        
        # Train individual models
        cpu_model = self.train_cpu_model(X_cpu, y_cpu_time)
        gpu_model = self.train_gpu_model(X_gpu, y_gpu_time)
        route_model = self.train_routing_model(X, y_device)
        
        # Generate visualizations
        self.generate_plots(df)
        
        # Save training metadata
        metadata = {
            'training_samples': len(df),
            'cpu_samples': len(X_cpu),
            'gpu_samples': len(X_gpu),
            'feature_columns': list(X.columns),
            'task_types': list(df['task_type'].unique()),
            'cpu_rmse': cpu_model['rmse'] if cpu_model else None,
            'gpu_rmse': gpu_model['rmse'] if gpu_model else None,
            'routing_accuracy': route_model['accuracy'] if route_model else None
        }
        
        with open(f'{self.models_dir}/training_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"\n=== Training Complete ===")
        print(f"Models saved to {self.models_dir}/")
        print(f"CPU Model RMSE: {metadata['cpu_rmse']:.3f} ms" if metadata['cpu_rmse'] else "CPU Model: Failed")
        print(f"GPU Model RMSE: {metadata['gpu_rmse']:.3f} ms" if metadata['gpu_rmse'] else "GPU Model: Failed")
        print(f"Routing Accuracy: {metadata['routing_accuracy']:.3f}" if metadata['routing_accuracy'] else "Routing Model: Failed")
        
        return metadata

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train HeteroSched ML models')
    parser.add_argument('--csv', default='logs/task_log.csv', help='Path to task log CSV')
    parser.add_argument('--models-dir', default='models', help='Directory to save models')
    parser.add_argument('--plot', action='store_true', help='Generate plots only')
    
    args = parser.parse_args()
    
    trainer = HeteroSchedMLTrainer(csv_path=args.csv, models_dir=args.models_dir)
    
    if args.plot:
        # Just generate plots from existing data
        _, _, _, _, _, _, df = trainer.load_and_preprocess_data()
        trainer.generate_plots(df)
    else:
        # Full training pipeline
        trainer.train_all_models()

if __name__ == '__main__':
    main()