import json
import os

# Define the directory path where the SMAC3 logs are stored
log_dir = "smac3_output/IoU0_#2/3GNN/62891c9069c5ee8cb2b1618137272748"
#e2ebb8e50955796f7388cd8c228cc757, e41b2a8c0ef33517ceddbc2a61e49dfd = 2LIN
#d16fe6b9dd18d8366d190724fb615cc4, c3a8e56b440378a5ee9162293d636e11
subdir = "0"  # Assuming there's only one subdirectory named "0"

# File paths
runhistory_path = os.path.join(log_dir, subdir, "runhistory.json")
configspace_path = os.path.join(log_dir, subdir, "configspace.json")
scenario_path = os.path.join(log_dir, subdir, "scenario.json")

# Function to load JSON files
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Load the JSON files
runhistory = load_json(runhistory_path)
configspace = load_json(configspace_path)
scenario = load_json(scenario_path)

arch_results = []

# Extract the best configuration
best_config = None
best_performance = float('inf')  # Assuming a minimization problem

for run in runhistory['data']:
    config_id = run[0]
    performance = run[4]
    # Check if this performance is better than the best found so far
    if performance < best_performance:
        best_performance = performance
        best_config = runhistory['configs'][str(config_id)]
    arch_results.append({'id':config_id, 'performance':1-best_performance})
    
print("Best Configuration:", best_config)
print("Best Performance:", 1-best_performance)

arch_results = sorted(arch_results, key=lambda x: x['performance'])
#print(arch_results[-5:])
for arch_result in arch_results:
    print(arch_results)
    print('\n')