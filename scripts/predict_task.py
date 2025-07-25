#!/usr/bin/env python3
"""
HeteroSched ML Prediction Interface

Simple script that can be called from C code to make ML predictions.
Usage: python predict_task.py <task_type> <task_size> <rows> <cols>
Returns: CPU|GPU <predicted_cpu_time> <predicted_gpu_time>
"""

import pickle
import numpy as np
import pandas as pd
import sys
import os

class HeteroSchedPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.cpu_model = None
        self.gpu_model = None
        self.route_model = None
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        try:
            with open(f'{self.models_dir}/model_cpu.pkl', 'rb') as f:
                self.cpu_model = pickle.load(f)
        except FileNotFoundError:
            print("Warning: CPU model not found", file=sys.stderr)
            
        try:
            with open(f'{self.models_dir}/model_gpu.pkl', 'rb') as f:
                self.gpu_model = pickle.load(f)
        except FileNotFoundError:
            print("Warning: GPU model not found", file=sys.stderr)
            
        try:
            with open(f'{self.models_dir}/model_route.pkl', 'rb') as f:
                self.route_model = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Routing model not found", file=sys.stderr)
            
    def create_features(self, task_type, task_size, rows, cols):
        """Create feature vector from task parameters"""
        # Log-scale features
        log_task_size = np.log1p(task_size)
        log_rows = np.log1p(rows)
        log_cols = np.log1p(cols)
        
        # Complexity score
        complexity_score = task_size * (rows if task_type == 'MATMUL' else 1)
        log_complexity = np.log1p(complexity_score)
        
        # Base features
        features = {
            'log_task_size': log_task_size,
            'log_rows': log_rows, 
            'log_cols': log_cols,
            'log_complexity': log_complexity
        }
        
        # One-hot encode task types
        task_types = ['MATMUL', 'RELU', 'VEC_ADD', 'VEC_SCALE']
        for tt in task_types:
            features[f'task_{tt}'] = 1.0 if task_type == tt else 0.0
            
        return features
        
    def predict_device(self, task_type, task_size, rows, cols):
        """Predict optimal device (CPU or GPU)"""
        if not self.route_model:
            # Fallback to static rules
            return self.static_fallback(task_type, task_size, rows, cols)
            
        features = self.create_features(task_type, task_size, rows, cols)
        feature_vector = np.array([list(features.values())])
        
        # Scale features
        feature_vector_scaled = self.route_model['scaler'].transform(feature_vector)
        
        # Make prediction
        prediction = self.route_model['model'].predict(feature_vector_scaled)[0]
        
        return prediction
        
    def predict_runtime(self, task_type, task_size, rows, cols, device):
        """Predict runtime for given device"""
        features = self.create_features(task_type, task_size, rows, cols)
        feature_vector = np.array([list(features.values())])
        
        if device == 'CPU' and self.cpu_model:
            feature_vector_scaled = self.cpu_model['scaler'].transform(feature_vector)
            runtime = self.cpu_model['model'].predict(feature_vector_scaled)[0]
            return max(0.01, runtime)  # Minimum 0.01ms
        elif device == 'GPU' and self.gpu_model:
            feature_vector_scaled = self.gpu_model['scaler'].transform(feature_vector)
            runtime = self.gpu_model['model'].predict(feature_vector_scaled)[0]
            return max(0.01, runtime)  # Minimum 0.01ms
        else:
            # Fallback estimates
            return self.static_runtime_fallback(task_type, task_size, rows, cols, device)
            
    def static_fallback(self, task_type, task_size, rows, cols):
        """Static fallback rules when ML models unavailable"""
        if task_type == 'VEC_ADD':
            return 'GPU' if task_size > 50000 else 'CPU'
        elif task_type == 'MATMUL':
            return 'GPU' if rows > 128 or cols > 128 else 'CPU'
        elif task_type == 'VEC_SCALE':
            return 'GPU' if task_size > 100000 else 'CPU'
        elif task_type == 'RELU':
            return 'GPU' if task_size > 10000 else 'CPU'
        else:
            return 'CPU'
            
    def static_runtime_fallback(self, task_type, task_size, rows, cols, device):
        """Static runtime estimates when ML models unavailable"""
        # Very rough estimates based on task complexity
        if task_type == 'VEC_ADD':
            base_time = task_size * 0.001  # 1μs per element
        elif task_type == 'MATMUL':
            base_time = rows * cols * rows * 0.01  # O(n^3) complexity
        elif task_type == 'VEC_SCALE':
            base_time = task_size * 0.0005  # 0.5μs per element
        elif task_type == 'RELU':
            base_time = task_size * 0.0002  # 0.2μs per element
        else:
            base_time = 1.0
            
        # Adjust for device (GPU typically faster for large tasks)
        if device == 'GPU' and task_size > 10000:
            base_time *= 0.3  # GPU is ~3x faster for large tasks
        elif device == 'CPU':
            base_time *= 1.0  # CPU baseline
            
        return max(0.01, base_time)

def main():
    """Command line interface for C integration"""
    if len(sys.argv) != 5:
        print("Usage: python predict_task.py <task_type> <task_size> <rows> <cols>")
        print("Returns: <device> <cpu_time> <gpu_time>")
        sys.exit(1)
        
    task_type = sys.argv[1]
    task_size = int(sys.argv[2])
    rows = int(sys.argv[3])
    cols = int(sys.argv[4])
    
    predictor = HeteroSchedPredictor()
    
    # Predict optimal device
    device = predictor.predict_device(task_type, task_size, rows, cols)
    
    # Predict runtimes for both devices
    cpu_time = predictor.predict_runtime(task_type, task_size, rows, cols, 'CPU')
    gpu_time = predictor.predict_runtime(task_type, task_size, rows, cols, 'GPU')
    
    # Output in format: <device> <cpu_time> <gpu_time>
    print(f"{device} {cpu_time:.4f} {gpu_time:.4f}")

if __name__ == '__main__':
    main()