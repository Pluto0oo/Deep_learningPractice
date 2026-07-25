import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess


def main():
    print("=" * 60)
    print("Running Full Reinforcement Learning Experiment")
    print("=" * 60)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(scripts_dir), 'configs', 'base.yaml')

    steps = [
        ("Step 1: Train DQN", os.path.join(scripts_dir, 'train_dqn.py')),
        ("Step 2: Collect Expert Data", os.path.join(scripts_dir, 'collect_expert_data.py')),
        ("Step 3: Train Behavioral Cloning", os.path.join(scripts_dir, 'train_bc.py')),
        ("Step 4: Evaluate Models", os.path.join(scripts_dir, 'evaluate.py'))
    ]

    for step_name, script_path in steps:
        print(f"\n{step_name}")
        print("-" * 40)
        
        cmd = [sys.executable, script_path, '--config', config_path]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        if result.returncode != 0:
            print(f"Error: {step_name} failed with exit code {result.returncode}")
            break

    print("\n" + "=" * 60)
    print("Experiment pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
