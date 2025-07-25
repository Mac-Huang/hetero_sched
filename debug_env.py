#!/usr/bin/env python3
"""Debug script to check environment state normalization"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from rl.environments.hetero_env import make_hetero_env
import numpy as np

def debug_normalization():
    env = make_hetero_env('default')
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Min value: {obs.min()}")
    print(f"Max value: {obs.max()}")
    print(f"Out of range indices: {np.where((obs < 0) | (obs > 1))[0]}")
    print(f"Out of range values: {obs[(obs < 0) | (obs > 1)]}")
    
    print("\nDetailed breakdown:")
    task_features = obs[:9]
    system_features = obs[9:20]
    queue_features = obs[20:28]
    performance_features = obs[28:36]
    
    print(f"Task features: min={task_features.min():.3f}, max={task_features.max():.3f}")
    print(f"System features: min={system_features.min():.3f}, max={system_features.max():.3f}")
    print(f"Queue features: min={queue_features.min():.3f}, max={queue_features.max():.3f}")  
    print(f"Performance features: min={performance_features.min():.3f}, max={performance_features.max():.3f}")
    
    # Check individual features
    if performance_features.max() > 1.0 or performance_features.min() < 0.0:
        print(f"Performance history values: {performance_features}")
    
    env.close()

if __name__ == '__main__':
    debug_normalization()