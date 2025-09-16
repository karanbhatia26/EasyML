import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from marl.agents.student import StudentAgent
from marl.agents.teacher import TeacherAgent
from marl.environments.pipeline_env import PipelineEnvironment
from marl.utils.credit_assignment import CreditAssignment
from marl.utils.component_guide import ComponentGuide
from marl.environments.ml_components import COMPONENT_MAP
from marl.utils.visualizer import PerformanceVisualizer, CollaborationVisualizer, TeacherContributionTracker
from sklearn.datasets import fetch_openml, fetch_covtype
from marl.models.double_dqn import DQNetwork
import copy
import polars as pl
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def set_seed(seed: int = 42):
    import random as _random
    import numpy as _np
    _np.random.seed(seed)
    _random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dataset_with_polars(dataset_name):
    print(f"Loading {dataset_name} dataset with Polars...")
    
    if (dataset_name == "adult"):
        data = fetch_openml(name='adult', version=2, as_frame=True)
        X = pl.from_pandas(data.data)
        y = pl.from_pandas(pd.DataFrame(data.target, columns=['target']))
    elif (dataset_name == "iris"):
        data = fetch_openml(name='iris', version=1, as_frame=True)
        X = pl.from_pandas(data.data)
        y = pl.from_pandas(pd.DataFrame(data.target, columns=['target']))
    elif (dataset_name == "digits"):
        data = fetch_openml(name='mnist_784', version=1, as_frame=True)
        X = pl.from_pandas(data.data)
        y = pl.from_pandas(pd.DataFrame(data.target, columns=['target']))
    elif (dataset_name == "covtype"):
        data = fetch_covtype(as_frame=True)
        X = pl.from_pandas(data.data)
        y = pl.from_pandas(pd.DataFrame(data.target, columns=['target']))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if (y.dtypes[0] == pl.Categorical or y.dtypes[0] == pl.Utf8):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_array = le.fit_transform(y.to_pandas().values.ravel())
        y = pl.from_numpy(y_array, schema=["target"])
    
    train_mask = pl.Series([True if i < int(0.8 * X.height) else False for i in range(X.height)])
    
    X_train = X.filter(train_mask)
    X_test = X.filter(~train_mask)
    y_train = y.filter(train_mask)
    y_test = y.filter(~val_mask)
    
    val_mask = pl.Series([True if i >= int(0.75 * X_train.height) else False for i in range(X_train.height)])
    X_val = X_train.filter(val_mask)
    X_train = X_train.filter(~val_mask)
    y_val = y_train.filter(val_mask)
    y_train = y_train.filter(~val_mask)
    
    return {
        'X_train': X_train.to_pandas(),
        'X_test': X_test.to_pandas(),
        'X_val': X_val.to_pandas(),
        'y_train': y_train.to_pandas().values.ravel(),
        'y_test': y_test.to_pandas().values.ravel(),
        'y_val': y_val.to_pandas().values.ravel(),
        'feature_names': X.columns,
        'n_classes': len(y.unique().to_numpy())
    }

def load_dataset(dataset_name):
    if dataset_name == "adult":
        print("Loading Adult dataset...")
        data = fetch_openml(name='adult', version=2, as_frame=True)
    elif dataset_name == "iris":
        print("Loading Iris dataset...")
        data = fetch_openml(name='iris', version=1, as_frame=True)
    elif dataset_name == "digits":
        print("Loading Digits dataset...")
        data = fetch_openml(name='mnist_784', version=1, as_frame=True)
    elif dataset_name == "covtype":
        print("Loading Covertype dataset...")
        data = fetch_covtype(as_frame=True)
    elif dataset_name == "credit-g":
        data = fetch_openml(data_id=31, as_frame=True)
    elif dataset_name == "travel":
        data = fetch_openml(data_id=45065, as_frame=True)
    elif dataset_name == "banknote":
        data = fetch_openml(name='BNG(banknote-authentication)', data_id=1462, as_frame=True)
    elif dataset_name == "click-prediction":
        data = fetch_openml(name='click_prediction_small', data_id=4134, as_frame=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    
    X = data.data.copy()
    y = data.target.copy()
    
    if y.dtype == 'object' or y.dtype.name == 'category':
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_val': X_val,
        'y_train': y_train, 
        'y_test': y_test,
        'y_val': y_val,
        'feature_names': X.columns,
        'n_classes': len(np.unique(y))
    }

def print_gpu_statistics():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Create a shared environment state monitor
class StateChangeMonitor:
    def __init__(self, initial_dim=None):
        self.current_dim = initial_dim
        self.change_listeners = []
        
    def register_agent(self, agent):
        self.change_listeners.append(agent)
        
    def check_state(self, state):
        if state is None:
            return
            
        new_dim = state.shape[0] if hasattr(state, 'shape') else len(state)
        if self.current_dim is None:
            self.current_dim = new_dim
            return False
            
        if new_dim != self.current_dim:
            print(f"STATE DIMENSION CHANGED: {self.current_dim} → {new_dim}")
            self.current_dim = new_dim
            
            # Notify all agents
            for agent in self.change_listeners:
                if hasattr(agent, 'rebuild_networks'):
                    agent.rebuild_networks(new_dim)
            
            return True
        return False

def main():
    print_gpu_statistics()
    dataset = load_dataset("iris")
    
    env = PipelineEnvironment(dataset)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    student_config = {
        'learning_rate': 1e-4,
        'epsilon': 1.0,
        'epsilon_min': 0.1
    }
    
    teacher_config = {
        'learning_rate': 5e-5,
        'epsilon': 0.5,
        'epsilon_min': 0.05
    }
    
    student = StudentAgent(state_dim, action_dim, student_config, device = DEVICE)
    teacher = TeacherAgent(state_dim, action_dim, teacher_config, device = DEVICE)
    
    # Initialize monitor
    state_monitor = StateChangeMonitor(state_dim)
    state_monitor.register_agent(student)
    state_monitor.register_agent(teacher)
    
    credit_assigner = CreditAssignment()
    
    episodes = 20
    
    all_rewards = []
    all_performances = []
    best_performance = 0
    best_pipeline = None
    
    for episode in range(episodes):
        state = env.reset()
        teacher_state = env.get_teacher_state([])  # Empty history for initial state
        teacher_state_dim = len(teacher_state)
        print(f"Student state dimension: {state_dim}, Teacher state dimension: {teacher_state_dim}")
        state_monitor.check_state(state)

        if state.shape[0] != state_dim:
            print(f"State dimension changed: {state_dim} -> {state.shape[0]}")
            state_dim = state.shape[0]
            student = StudentAgent(state_dim, action_dim, student_config)
            teacher = TeacherAgent(teacher_state_dim, action_dim, teacher_config)
        
        done = False
        episode_reward = 0
        pipeline_components = []
        
        while not done:
            valid_actions = env.get_filtered_actions()
            print(f"  Valid actions: {valid_actions}")

            if not valid_actions:
                print("  No valid actions available - pipeline complete")
                done = True
                continue

            action = student.act(state, valid_actions)

            if action == -1:
                print("  Agent returned no valid action - pipeline complete")
                done = True
                continue

            teacher_feedback = teacher.act(state, valid_actions)
            should_intervene = False 
            if np.random.rand() < 0.3:
                action = teacher_feedback
                
            next_state, reward, done, info = env.step(action)
            performance = info["performance"]
            
            component = env.available_components[action]
            pipeline_components.append(component)
            
            teacher.learn(state, action, reward, next_state, done)
            
            student.learn(state, action, reward, next_state, done, teacher_intervened=should_intervene)
            
            episode_reward += reward
            state = next_state
        
        if len(pipeline_components) > 1:
            def evaluate_mod_pipeline(mod_pipeline):
                return performance * 0.9
            
            component_credits = credit_assigner.ablation_credit(
                pipeline_components, performance, evaluate_mod_pipeline)
            
            credit_assigner.update_component_credits(component_credits, performance)
        
        if done:
            credits = credit_assigner.assign_marl_credit(...)
            
            # Adjust final rewards based on credit assignment
            student_final_reward = performance * credits['student_credit'] 
            teacher_final_reward = performance * credits['teacher_credit']
            
            # Apply these final rewards to recent experiences
            student.apply_final_reward(student_final_reward, decay=0.95)
            teacher.apply_final_reward(teacher_final_reward, decay=0.95)
        
        all_rewards.append(episode_reward)
        all_performances.append(performance)
        
        if performance > best_performance:
            best_performance = performance
            best_pipeline = pipeline_components.copy()
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_performance = np.mean(all_performances[-10:])
            print(f"Episode: {episode+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Avg Performance: {avg_performance:.4f}")
            print(f"Current pipeline: {pipeline_components}")

    print("\n=== Training Complete ===")
    print(f"Best performance: {best_performance:.4f}")
    print(f"Best pipeline: {best_pipeline}")
    os.makedirs("models", exist_ok=True)
    # Save to distinct, valid filenames
    student.save("models/student_model_basic_iris.pt")
    teacher.save("models/teacher_model_basic_iris.pt")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(all_performances)
    plt.title('Pipeline Performance')
    plt.xlabel('Episode')
    plt.ylabel('Performance')
    
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()

def marl_training(dataset_name="iris", episodes=20):
    set_seed(42)
    print_gpu_statistics()
    dataset = load_dataset(dataset_name)
    
    env = PipelineEnvironment(dataset, available_components=list(COMPONENT_MAP.keys()), max_pipeline_length=8, debug=False)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    teacher_state_dim = state_dim + action_dim
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    student_path = os.path.join(model_dir, f"student_model_marl_{dataset_name}.pt")
    teacher_path = os.path.join(model_dir, f"teacher_model_marl_{dataset_name}.pt")
    
    student_config = {
        'learning_rate': 1e-3,
        'epsilon': 1.0,
        'epsilon_min': 0.1
    }
    
    teacher_config = {
        'learning_rate': 1e-4, 
        'gamma': 0.99,
        'epsilon': 0.3,
        'intervention_decay': True,
        'batch_size': 32
    }
    state = env.reset()
    teacher_state = env.get_teacher_state([])
    teacher_state_dim = len(teacher_state)
    print(f"Student state dimension: {state_dim}, Teacher state dimension: {teacher_state_dim}")
    student = StudentAgent(state_dim, action_dim, student_config)
    teacher = TeacherAgent(teacher_state_dim, action_dim, teacher_config)
    if os.path.exists(student_path):
        try:
            student.load(student_path)
        except:
            print("Could not load student model - using new model")
    
    if os.path.exists(teacher_path):
        try:
            teacher.load(teacher_path)
        except:
            print("Could not load teacher model - using new model")
    print("\nAvailable components and their action indices:")
    for idx, component in enumerate(env.available_components):
        print(f"  {idx}: {component}")
    credit_assigner = CreditAssignment()
    visualizer = PerformanceVisualizer()
    contribution_tracker = TeacherContributionTracker(episodes)
    
    all_rewards = []
    all_performances = []
    best_performance = 0
    best_pipeline = None
    
    # Initialize pipeline memory
    pipeline_memory = []
    best_pipelines = []
    
    print(f"Starting MARL training with {episodes} episodes")
    
    for episode in range(episodes):
        state = env.reset()

        if state.shape[0] != state_dim:
            print(f"State dimension changed: {state_dim} -> {state.shape[0]}")
            state_dim = state.shape[0]
            
            # Get the new teacher state dimension
            teacher_state = env.get_teacher_state([])
            teacher_state_dim = len(teacher_state)
            print(f"New student state dimension: {state_dim}, New teacher state dimension: {teacher_state_dim}")
            
            print(f"Recreating agents with updated dimensions")
            student = StudentAgent(state_dim, action_dim, student_config)
            teacher = TeacherAgent(teacher_state_dim, action_dim, teacher_config)
        
        done = False
        episode_reward = 0
        pipeline_components = []
        
        # Track agent actions and interactions
        student_actions = []
        teacher_interventions = []
        action_sources = []
        
        print(f"\nEpisode {episode+1}/{episodes}")
        
        student_history = []  # Track student's recent actions
        
        # Initialize counter for repetitive behavior
        repetitive_action_count = 0
        max_repetitive_actions = 10
        last_valid_actions = None
        
        while not done:
            # Get valid actions
            valid_actions = env.get_filtered_actions()
            
            # Add this safety check
            if not valid_actions:
                print("WARNING: No valid actions available. Forcing END_PIPELINE.")
                # Find END_PIPELINE action
                for action, component in enumerate(env.available_components):
                    if str(component) == "END_PIPELINE":
                        final_action = action
                        break
                else:
                    # If END_PIPELINE not found, just pick action 0
                    print("ERROR: END_PIPELINE not found in available components!")
                    final_action = 0
            else:
                # Normal action selection logic
                student_action = student.act(state, valid_actions, env=env) 
            
            # Check for repetitive behavior
            if valid_actions == last_valid_actions:
                repetitive_action_count += 1
            else:
                repetitive_action_count = 0
            
            last_valid_actions = valid_actions.copy() if valid_actions else None
            
            # Early termination if we're in a cycle
            if repetitive_action_count > max_repetitive_actions:
                print("Detected action cycle - ending episode early")
                break
                
            # Student selects action
            student_action = student.act(state, valid_actions, env=env)  # Need to pass env
            if student_action == -1:
                print("Student returned invalid action - ending episode")
                break
                
            student_history.append(student_action)
                
            # Teacher decides whether to intervene
            teacher_state = env.get_teacher_state(student_history)
            should_intervene, teacher_action = teacher.act(
                teacher_state, valid_actions, student_action, env=env)
            
            # Process intervention
            final_action, action_source = env.process_teacher_intervention(
                student_action, should_intervene, teacher_action)
            
            # Take the action
            next_state, reward, done, info = env.step(final_action)
            
            # Track action for credit assignment
            student_actions.append(student_action)
            teacher_interventions.append(should_intervene)
            action_sources.append(action_source)
            
            # Add to pipeline
            component = env.available_components[final_action]
            pipeline_components.append(component)
            print(f"  Added {component} (from {action_source}), reward: {reward:.4f}")
            
            # Calculate agent-specific rewards
            student_reward = reward  # Base reward
            teacher_reward = env.calculate_teacher_reward(
                student_action, should_intervene, teacher_action, 
                info.get("performance", 0))
            
            # Store experiences
            student.learn(state, final_action, student_reward, next_state, done)
            teacher.learn(teacher_state, (should_intervene, teacher_action), 
                          teacher_reward, next_state, done)
            
            # Update trackers
            episode_reward += reward
            contribution_tracker.record_action(
                episode, student_action, teacher_action, 
                should_intervene, reward)
            
            # Move to next state
            state = next_state
            
        # Episode complete
        performance = info.get("performance", 0)
        
        # Final reward distribution
        if len(pipeline_components) > 1:
            # Estimate what performance would be without teacher
            student_only_performance = performance * 0.8  # Estimate
            
            # Calculate final credit assignment
            credits = credit_assigner.assign_marl_credit(
                pipeline_components, student_actions, teacher_interventions,
                performance, student_only_performance)
            
            # Print credit assignment
            print(f"  Credit assignment: Student {credits['student_credit']:.2f}, "
                  f"Teacher {credits['teacher_credit']:.2f}")
            
            # Update component knowledge
            teacher.update_component_knowledge(pipeline_components, performance)
        
        # Only for successful pipelines with multiple components
        if len(pipeline_components) > 1 and performance > 0:
            # Throttle expensive ablations and use stricter timeout
            if (episode + 1) % 5 == 0 and performance >= 0.1:
                component_credits = credit_assigner.assign_component_credit(
                    pipeline_components, performance,
                    lambda pipeline: env.evaluate_with_timeout(pipeline, timeout=120))
            else:
                component_credits = {}
                
            # Print component contributions
            print("\nComponent contributions:")
            for component, credit in sorted(component_credits.items(), key=lambda x: x[1], reverse=True):
                print(f"  {component}: {credit:.3f}")
                
            # Convert to agent credits
            credits = credit_assigner.translate_component_to_agent_credit(
                component_credits, pipeline_components, action_sources)
                
            # Use these credits for your existing credit assignment display
            student_credit = credits['student_credit']
            teacher_credit = credits['teacher_credit']
            
            print(f"  Credit assignment: Student {student_credit:.2f}, Teacher {teacher_credit:.2f}")
        
        if done:
            # Get proper credit assignment
            credits = credit_assigner.assign_marl_credit(
                pipeline_components, student_actions, teacher_interventions,
                performance)
                
            # Apply final rewards weighted by credit
            if len(pipeline_components) > 1:
                student.apply_final_reward(performance * credits['student_credit'])
                teacher.apply_final_reward(performance * credits['teacher_credit'])
                
            # Update environment's pipeline memory with successful pipelines
            if performance > 0.7:
                env._update_pipeline_memory(pipeline_components, performance)
        
        visualizer.add_episode_data(episode_reward, performance, 
                                   pipeline_components, any(teacher_interventions))
        
        all_rewards.append(episode_reward)
        all_performances.append(performance)
        
        if performance > best_performance:
            best_performance = performance
            best_pipeline = pipeline_components.copy()
            
        print(f"Episode {episode+1} complete")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Performance: {performance:.4f}")
        print(f"  Teacher interventions: {sum(teacher_interventions)}/{len(teacher_interventions)}"
              f" ({sum(teacher_interventions)/max(1, len(teacher_interventions)):.1%})")
        
        # Save periodically
        if (episode + 1) % 5 == 0:
            student.save(student_path)
            teacher.save(teacher_path)
            
        contribution_tracker.record_episode_performance(episode, performance)
        
        # After each episode completes:
        if 'performance' in info and info['performance'] > 0.7:
            # Remember successful pipelines
            pipeline_memory.append({
                'pipeline': env.current_pipeline.copy(),
                'performance': info['performance'],
                'episode': episode
            })
            
            # Sort by performance
            pipeline_memory.sort(key=lambda x: x['performance'], reverse=True)
            
            # Keep top 5
            pipeline_memory = pipeline_memory[:5]
            
            # Print memory
            print("Pipeline memory updated:")
            for i, p in enumerate(pipeline_memory):
                print(f"  #{i+1}: {[str(c) for c in p['pipeline']]} - {p['performance']:.4f}")
        
        # Analyze teacher interventions and adjust epsilon
        if hasattr(teacher, 'intervention_outcomes') and len(teacher.intervention_outcomes) > 0:
            teacher._analyze_interventions()
            print(f"Teacher epsilon adjusted to: {teacher.config['epsilon']:.2f}")
    
    # Save final models
    student.save(student_path)
    teacher.save(teacher_path)
    
    print("\n=== MARL Training Complete ===")
    print(f"Best performance: {best_performance:.4f}")
    print(f"Best pipeline: {best_pipeline}")
    
    # Generate visualizations
    visualizer.plot_learning_curves(window_size=5, save_path=f"marl_learning_curves_{dataset_name}.png")
    visualizer.plot_pipeline_evolution(save_path=f"marl_pipeline_evolution_{dataset_name}.png")
    contribution_tracker.print_contribution_report()
    contribution_tracker.plot_teacher_contribution(save_path=f"marl_teacher_contribution_{dataset_name}.png")
    env.print_pipeline_statistics()
    
    plot_intervention_rate(teacher, dataset_name)
    
    return env

def test_run(dataset="iris", episodes=500):
    return marl_training(dataset_name=dataset, episodes=episodes)

def adapt_model_to_new_dimensions(saved_model_path, new_input_dim, new_output_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(saved_model_path, map_location=device)
    new_model = DQNetwork(new_input_dim, new_output_dim).to(device)
    
    old_state_dict = checkpoint['policy_net']
    new_state_dict = new_model.state_dict()
    
    # Copy matching parameters
    for name, param in new_state_dict.items():
        if name in old_state_dict and param.shape == old_state_dict[name].shape:
            new_state_dict[name] = old_state_dict[name]

    new_model.load_state_dict(new_state_dict)
    return new_model

def plot_intervention_rate(teacher,dataset_name):
    """Plot teacher intervention rate over episodes"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Extract data from intervention history
    episodes = [x[0] for x in teacher.intervention_history]
    interventions = [1 if x[1] else 0 for x in teacher.intervention_history]
    
    # Calculate moving average
    window = 50
    smoothed = []
    for i in range(len(episodes)):
        start_idx = max(0, i - window + 1)
        smoothed.append(sum(interventions[start_idx:i+1]) / (i - start_idx + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, interventions, 'o', alpha=0.3, label='Interventions')
    plt.plot(episodes, smoothed, 'r-', label=f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Intervention Rate')
    plt.title('Teacher Intervention Over Training')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"intervention_rate_{dataset_name}.png")
    plt.close()

# if __name__ == "__main__":
#      def test_run(dataset="iris", episodes=500):
#     print(f"Running test with dataset={dataset}, episodes={int(episodes)}")
#     return marl_training(dataset_name=dataset, episodes=int(episodes))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train MARL agents for AutoML")
    parser.add_argument("--dataset", type=str, default="iris", help="Dataset name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    
    args = parser.parse_args()
    test_run(dataset=args.dataset, episodes=args.episodes)