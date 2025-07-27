
"""
Evaluation wrapper for SampleDeepRL
Generated automatically by HeteroSched Benchmark Framework
"""

import sys
import json
import time
import traceback
from algorithm import SchedulingAlgorithm

def main():
    """Main evaluation function"""
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <input_file> <output_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Load input data
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Initialize algorithm
        algorithm = SchedulingAlgorithm()
        
        # Run evaluation
        start_time = time.time()
        results = algorithm.schedule(data)
        end_time = time.time()
        
        # Save results
        output_data = {
            'results': results,
            'execution_time': end_time - start_time,
            'metadata': {
                'algorithm': 'SampleDeepRL',
                'version': '1.0.0',
                'timestamp': time.time()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print("Evaluation completed successfully")
        
    except Exception as e:
        error_data = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(error_data, f, indent=2)
            
        print(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
