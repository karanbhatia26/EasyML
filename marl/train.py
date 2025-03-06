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
from marl.utils.visualizer import PerformanceVisualizer, CollaborationVisualizer, TeacherContributionTracker
from sklearn.datasets import fetch_openml, fetch_covtype
from marl.models.double_dqn import DQNetwork
import copy
import polars as pl

def load_dataset_with_polars(dataset_name):
    """Dataset loader using Polars for better performance."""
    print(f"Loading {dataset_name} dataset with Polars...")
    
    if (dataset_name == "adult"):
        data = fetch_openml(name='adult', version=2, as_frame=True)
        # Convert to polars
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
        # Convert to polars - this will be much faster with Polars
        X = pl.from_pandas(data.data)
        y = pl.from_pandas(pd.DataFrame(data.target, columns=['target']))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert target to numeric if needed
    if (y.dtypes[0] == pl.Categorical or y.dtypes[0] == pl.Utf8):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_array = le.fit_transform(y.to_pandas().values.ravel())
        y = pl.from_numpy(y_array, schema=["target"])
    
    # Use polars' built-in train_test_split functionality
    train_mask = pl.Series([True if i < int(0.8 * X.height) else False for i in range(X.height)])
    
    # Split X and y using the same mask
    X_train = X.filter(train_mask)
    X_test = X.filter(~train_mask)
    y_train = y.filter(train_mask)
    y_test = y.filter(~train_mask)
    
    # Further split train into train/val
    val_mask = pl.Series([True if i >= int(0.75 * X_train.height) else False for i in range(X_train.height)])
    X_val = X_train.filter(val_mask)
    X_train = X_train.filter(~val_mask)
    y_val = y_train.filter(val_mask)
    y_train = y_train.filter(~val_mask)
    
    # Convert back to pandas for scikit-learn compatibility
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
    """Generic dataset loader that works for any dataset."""
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
        data = fetch_covtype(as_frame=True) # This one is a bit larger
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    
    X = data.data.copy()
    y = data.target.copy()
    
    # Convert target to numeric if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)
    
    # Create train/val/test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Return a dictionary
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

def main():
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
    
    student = StudentAgent(state_dim, action_dim, student_config)
    teacher = TeacherAgent(state_dim, action_dim, teacher_config)
    
    credit_assigner = CreditAssignment()
    
    episodes = 10
    
    all_rewards = []
    all_performances = []
    best_performance = 0
    best_pipeline = None
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        pipeline_components = []
        
        while not done:
            valid_actions = env.get_valid_actions()
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
            if np.random.rand() < 0.3:
                action = teacher_feedback
                
            next_state, reward, done, info = env.step(action)
            performance = info["performance"]
            
            component = env.available_components[action]
            pipeline_components.append(component)
            
            teacher.learn(state, action, reward, next_state, done)
            
            student.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        if len(pipeline_components) > 1:
            def evaluate_mod_pipeline(mod_pipeline):
                return performance * 0.9
            
            component_credits = credit_assigner.ablation_credit(
                pipeline_components, performance, evaluate_mod_pipeline)
            
            credit_assigner.update_component_credits(component_credits, performance)
        
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
    student.save("models/student_model.pt")
    teacher.save("models/teacher_model.pt")
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
def test_run():
    # Change this to a more complex dataset
    dataset = load_dataset("covtype")  # or "adult"
    
    # Increase the maximum pipeline length
    env = PipelineEnvironment(dataset, max_pipeline_length=6)  # Allow up to 6 components
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create the model directory
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    student_path = os.path.join(model_dir, "student_model_test.pt")
    teacher_path = os.path.join(model_dir, "teacher_model_test.pt")
    
    student_config = {
        'learning_rate': 1e-3,
        'epsilon': 1.0,
        'epsilon_min': 0.5
    }
    
    teacher_config = {
        'learning_rate': 5e-4,
        'epsilon': 0.7,
        'epsilon_min': 0.3
    }
    
    student = StudentAgent(state_dim, action_dim, student_config)
    teacher = TeacherAgent(state_dim, action_dim, teacher_config)
    
    # Try to load existing models with adaptation if needed
    if os.path.exists(student_path):
        print("Loading existing student model...")
        try:
            student.load(student_path)
            print("Student model loaded successfully")
        except RuntimeError as e:
            print(f"Dimension mismatch in student model: {e}")
            print("Adapting model to new dimensions...")
            
            # Create adapted model
            adapted_model = adapt_model_to_new_dimensions(
                student_path, 
                state_dim,  # New input dimension
                action_dim  # New output dimension
            )
            
            # Replace student's model with adapted version
            student.model.policy_net = adapted_model
            student.model.target_net = copy.deepcopy(adapted_model)
            print("Model successfully adapted to new dimensions")

    if os.path.exists(teacher_path):
        print("Loading existing teacher model...")
        try:
            teacher.load(teacher_path)
            print("Teacher model loaded successfully")
        except RuntimeError as e:
            print(f"Dimension mismatch in teacher model: {e}")
            print("Adapting model to new dimensions...")
            
            # Create adapted model
            adapted_model = adapt_model_to_new_dimensions(
                teacher_path, 
                state_dim,  # New input dimension
                action_dim  # New output dimension
            )
            
            # Replace teacher's model with adapted version
            teacher.model.policy_net = adapted_model
            teacher.model.target_net = copy.deepcopy(adapted_model)
            print("Model successfully adapted to new dimensions")
    
    credit_assigner = CreditAssignment()
    component_guide = ComponentGuide()
    visualizer = PerformanceVisualizer()
    episodes = 100
    
    all_rewards = []
    all_performances = []
    best_performance = 0
    best_pipeline = None
    
    print("Starting test run with", episodes, "episodes")
    
    # Initialize the visualizer and contribution tracker
    collab_viz = CollaborationVisualizer()
    contribution_tracker = TeacherContributionTracker(episodes)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        pipeline_components = []
        teacher_used_this_episode = False
        
        print(f"\nEpisode {episode+1}/{episodes}")
        
        while not done:
            valid_actions = env.get_valid_actions()
            teacher_action = teacher.act(state, valid_actions)
            student_action = student.act(state, valid_actions)
           
            # Use teacher's suggestion with decreasing probability
            use_teacher = np.random.rand() < 0.3
            if use_teacher:
                action = teacher_action
                teacher_used_this_episode = True
            else:
                action = student_action
            
            next_state, reward, done, info = env.step(action)
            performance = info["performance"]
            
            component = env.available_components[action]
            pipeline_components.append(component)
            
            print(f"  Added component: {component}, reward: {reward:.4f}")
            
            teacher.learn(state, action, reward, next_state, done)
            student.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            # Record the interaction
            collab_viz.record_interaction(student_action, teacher_action, use_teacher, reward)
            
            # Record for teacher contribution tracking
            contribution_tracker.record_action(
                episode, student_action, teacher_action, use_teacher, reward)
        
        component_guide.update(pipeline_components, performance)
        
        visualizer.add_episode_data(episode_reward, performance, 
                                   pipeline_components, teacher_used_this_episode)
        
        if len(pipeline_components) > 1:
            def evaluate_mod_pipeline(mod_pipeline):
                return performance * 0.9
            
            component_credits = credit_assigner.ablation_credit(
                pipeline_components, performance, evaluate_mod_pipeline)
            
            credit_assigner.update_component_credits(component_credits, performance)
        
        # Track metrics
        all_rewards.append(episode_reward)
        all_performances.append(performance)
        
        if performance > best_performance:
            best_performance = performance
            best_pipeline = pipeline_components.copy()
        
        # Print episode summary
        print(f"Episode {episode+1} complete")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Performance: {performance:.4f}")
        print(f"  Pipeline: {pipeline_components}")
        
        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Saving checkpoint at episode {episode+1}")
            student.save(student_path)
            teacher.save(teacher_path)
        
        # Record episode performance
        contribution_tracker.record_episode_performance(episode, performance)
    
    # Final save at the end of all episodes
    print("Saving final models...")
    student.save(student_path)
    teacher.save(teacher_path)
    
    print("\n=== Test Run Complete ===")
    print(f"Best performance: {best_performance:.4f}")
    print(f"Best pipeline: {best_pipeline}")
    visualizer.plot_learning_curves(window_size=5, save_path="learning_curves_detailed.png")
    visualizer.plot_pipeline_evolution(save_path="pipeline_evolution.png")
    
    print("\nTop Component Transitions:")
    for comp in component_guide.transition_scores:
        transitions = component_guide.get_transition_preference(comp)
        if transitions:
            top_next = max(transitions.items(), key=lambda x: x[1])
            print(f"  {comp} → {top_next[0]}: {top_next[1]:.4f}")

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
    plt.savefig("test_learning_curves.png")
    plt.show()
    
    # At the end of test_run
    collab_viz.plot_collaboration(save_path="agent_collaboration.png")
    
    # At the end of test_run, print the contribution report
    contribution_tracker.print_contribution_report()
    contribution_tracker.plot_teacher_contribution(save_path="teacher_contribution_analysis.png")
def adapt_model_to_new_dimensions(saved_model_path, new_input_dim, new_output_dim):
    """Load a model and adapt it to new input/output dimensions."""
    # Load the saved model state dict
    checkpoint = torch.load(saved_model_path)
    
    # Create a new model with the new dimensions
    new_model = DQNetwork(new_input_dim, new_output_dim)
    
    # Copy the weights that match, ignore the rest
    old_state_dict = checkpoint['policy_net']
    new_state_dict = new_model.state_dict()
    
    # Copy matching parameters
    for name, param in new_state_dict.items():
        if name in old_state_dict and param.shape == old_state_dict[name].shape:
            new_state_dict[name] = old_state_dict[name]
    
    # Load the updated state dict
    new_model.load_state_dict(new_state_dict)
    return new_model

if __name__ == "__main__":
    test_run()