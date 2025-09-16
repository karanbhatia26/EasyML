import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class PerformanceVisualizer:
    def __init__(self):
        self.rewards = []
        self.performances = []
        self.teacher_influence = []  # Track when teacher suggestions are used
        self.pipeline_lengths = []
        self.component_counts = {}
        self.best_pipelines = []
        
    def add_episode_data(self, reward, performance, pipeline, teacher_used=False):
        self.rewards.append(reward)
        self.performances.append(performance)
        self.teacher_influence.append(1 if teacher_used else 0)
        self.pipeline_lengths.append(len(pipeline))
        
        for component in pipeline:
            if component not in self.component_counts:
                self.component_counts[component] = 0
            self.component_counts[component] += 1
            
        # Track best pipelines
        if not self.best_pipelines or performance > self.performances[self.best_pipelines[-1]]:
            self.best_pipelines.append(len(self.performances) - 1)
    
    def plot_learning_curves(self, window_size=10, save_path=None):
        plt.figure(figsize=(15, 10))
        smoothed_rewards = self._moving_average(self.rewards, window_size)
        smoothed_perf = self._moving_average(self.performances, window_size)
        episodes = list(range(len(self.rewards)))
        
        # rewards
        plt.subplot(2, 2, 1)
        plt.plot(episodes, self.rewards, 'b-', alpha=0.3)
        plt.plot(episodes[window_size-1:], smoothed_rewards, 'b-', linewidth=2)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # performance
        plt.subplot(2, 2, 2)
        plt.plot(episodes, self.performances, 'r-', alpha=0.3)
        plt.plot(episodes[window_size-1:], smoothed_perf, 'r-', linewidth=2)
        plt.title('Pipeline Performance')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        
        # Plot teacher influence over time
        plt.subplot(2, 2, 3)
        teacher_inf_avg = self._moving_average(self.teacher_influence, window_size)
        plt.plot(episodes[window_size-1:], teacher_inf_avg, 'g-', linewidth=2)
        plt.title('Teacher Influence (Moving Average)')
        plt.xlabel('Episode')
        plt.ylabel('Teacher Advice Usage Rate')
        plt.ylim(0, 1)
        
        plt.subplot(2, 2, 4)
        top_components = sorted(self.component_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:8]
        names = [comp[:15] + '...' if len(comp) > 15 else comp for comp, _ in top_components]
        counts = [count for _, count in top_components]
        plt.barh(names, counts)
        plt.title('Most Used Components')
        plt.xlabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_pipeline_evolution(self, save_path=None):
        """Visualize pipeline length and composition evolution"""
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.pipeline_lengths, 'k-')
        plt.title('Pipeline Length Evolution')
        plt.xlabel('Episode')
        plt.ylabel('Pipeline Length')
        
        for best_idx in self.best_pipelines:
            plt.axvline(x=best_idx, color='r', linestyle='--', alpha=0.5)
        
        # Scatter plot of performance vs pipeline length
        plt.subplot(2, 1, 2)
        plt.scatter(self.pipeline_lengths, self.performances)
        plt.title('Performance vs Pipeline Length')
        plt.xlabel('Pipeline Length')
        plt.ylabel('Performance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _moving_average(self, data, window_size):
        """Calculate moving average with specified window size"""
        return [np.mean(data[i:i+window_size]) 
                for i in range(len(data) - window_size + 1)]

class CollaborationVisualizer:
    def __init__(self):
        self.teacher_suggestions = []
        self.student_choices = []
        self.agreement_rate = []
        self.teacher_reward = []
        self.student_reward = []
        
    def record_interaction(self, student_action, teacher_action, used_teacher, reward):
        self.student_choices.append(student_action)
        self.teacher_suggestions.append(teacher_action)
        self.agreement_rate.append(1 if student_action == teacher_action else 0)
        if used_teacher:
            self.teacher_reward.append(reward)
        else:
            self.student_reward.append(reward)
    
    def plot_collaboration(self, save_path=None):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        window = min(20, len(self.agreement_rate))
        if window > 0:
            agreement_avg = [sum(self.agreement_rate[max(0, i-window):i+1]) / 
                            min(window, i+1) for i in range(len(self.agreement_rate))]
            plt.plot(agreement_avg)
            plt.title('Student-Teacher Agreement Rate')
            plt.xlabel('Episode')
            plt.ylabel('Agreement %')
            plt.ylim(0, 1)
        
        # Compare reward distributions
        plt.subplot(2, 2, 2)
        plt.boxplot([self.student_reward, self.teacher_reward], labels=['Student', 'Teacher'])
        plt.title('Reward Comparison')
        plt.ylabel('Reward')
        plt.subplot(2, 2, 3)
        if len(self.agreement_rate) > 10:
            plt.hist(self.agreement_rate, bins=10, alpha=0.7)
            plt.title('Agreement Distribution')
            plt.xlabel('Agreement (1=yes, 0=no)')
            plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        if len(self.student_choices) > 10 and len(self.teacher_suggestions) > 10:
            unique_actions = len(set(self.student_choices + self.teacher_suggestions))
            student_dist = [self.student_choices.count(i)/len(self.student_choices) 
                          for i in range(unique_actions)]
            teacher_dist = [self.teacher_suggestions.count(i)/len(self.teacher_suggestions) 
                          for i in range(unique_actions)]
            plt.bar(range(unique_actions), student_dist, alpha=0.5, label='Student')
            plt.bar(range(unique_actions), teacher_dist, alpha=0.5, label='Teacher')
            plt.title('Action Distribution Comparison')
            plt.xlabel('Action Index')
            plt.ylabel('Selection Frequency')
            plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class TeacherContributionTracker:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        self.episode_bins = 10
        # Ensure bin_size is at least 1 to avoid division by zero for small runs
        self.bin_size = max(1, num_episodes // self.episode_bins)
        
        # Initialize counters
        self.teacher_actions_used = [0] * self.episode_bins
        self.total_actions = [0] * self.episode_bins
        self.teacher_rewards = [[] for _ in range(self.episode_bins)]
        self.student_rewards = [[] for _ in range(self.episode_bins)]
        self.bin_performances = [[] for _ in range(self.episode_bins)]
        
    def record_action(self, episode, student_action, teacher_action, used_teacher, reward):
        bin_idx = min(episode // self.bin_size, self.episode_bins - 1)
        
        self.total_actions[bin_idx] += 1
        if used_teacher:
            self.teacher_actions_used[bin_idx] += 1
            self.teacher_rewards[bin_idx].append(reward)
        else:
            self.student_rewards[bin_idx].append(reward)
            
    def record_episode_performance(self, episode, performance):
        bin_idx = min(episode // self.bin_size, self.episode_bins - 1)
        self.bin_performances[bin_idx].append(performance)
            
    def get_contribution_stats(self):
        contribution_rates = []
        for i in range(self.episode_bins):
            if self.total_actions[i] > 0:
                contribution_rates.append(self.teacher_actions_used[i] / self.total_actions[i])
            else:
                contribution_rates.append(0)
                
        avg_teacher_rewards = []
        for rewards in self.teacher_rewards:
            avg_teacher_rewards.append(np.mean(rewards) if rewards else 0)
            
        avg_student_rewards = []
        for rewards in self.student_rewards:
            avg_student_rewards.append(np.mean(rewards) if rewards else 0)
            
        avg_performances = []
        for perfs in self.bin_performances:
            avg_performances.append(np.mean(perfs) if perfs else 0)
            
        return {
            "contribution_rates": contribution_rates,
            "avg_teacher_rewards": avg_teacher_rewards,
            "avg_student_rewards": avg_student_rewards,
            "avg_performances": avg_performances,
            "bin_edges": [i * self.bin_size for i in range(self.episode_bins + 1)]
        }
        
    def print_contribution_report(self):
        stats = self.get_contribution_stats()
        
        print("\n=== Teacher Contribution Report ===")
        print("Episode Range | Teacher Usage | Teacher Reward | Student Reward | Performance")
        print("-" * 75)
        
        for i in range(self.episode_bins):
            start_ep = i * self.bin_size
            end_ep = min((i + 1) * self.bin_size - 1, self.num_episodes - 1)
            
            tr = stats['avg_teacher_rewards'][i]
            sr = stats['avg_student_rewards'][i]
            perf = stats['avg_performances'][i]
            tr_str = f"{tr:6.3f}" if self.teacher_rewards[i] else "   N/A"
            sr_str = f"{sr:6.3f}" if self.student_rewards[i] else "   N/A"
            perf_str = f"{perf:6.3f}" if self.bin_performances[i] else "   N/A"
            print(f"{start_ep:4d}-{end_ep:<4d}    | " +
                  f"{stats['contribution_rates'][i]*100:6.1f}%      | " +
                  f"{tr_str}       | " +
                  f"{sr_str}       | " +
                  f"{perf_str}")
                  
    def plot_teacher_contribution(self, save_path=None):
        stats = self.get_contribution_stats()
        bin_labels = [f"{i*self.bin_size}-{min((i+1)*self.bin_size-1, self.num_episodes-1)}" 
                     for i in range(self.episode_bins)]
        
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        plt.bar(bin_labels, [rate * 100 for rate in stats["contribution_rates"]])
        plt.title("Teacher Action Usage Rate")
        plt.xlabel("Episodes")
        plt.ylabel("Usage %")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        x = range(len(bin_labels))
        width = 0.35
        # Use zeros for empty bins to avoid NaN plotting issues
        tvals = [v if np.isfinite(v) else 0 for v in stats["avg_teacher_rewards"]]
        svals = [v if np.isfinite(v) else 0 for v in stats["avg_student_rewards"]]
        plt.bar([i-width/2 for i in x], tvals, width, label="Teacher")
        plt.bar([i+width/2 for i in x], svals, width, label="Student")
        plt.title("Average Reward by Agent")
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.xticks(x, bin_labels, rotation=45)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        ax1 = plt.gca()
        pvals = [v if np.isfinite(v) else 0 for v in stats["avg_performances"]]
        ax1.bar(bin_labels, pvals, alpha=0.7)
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Performance", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title("Performance vs Teacher Influence")
        ax1.set_xticklabels(bin_labels, rotation=45)
        
        ax2 = ax1.twinx()
        ax2.plot(bin_labels, [rate * 100 for rate in stats["contribution_rates"]], 
                'r-', linewidth=2, marker='o')
        ax2.set_ylabel("Teacher Usage %", color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()