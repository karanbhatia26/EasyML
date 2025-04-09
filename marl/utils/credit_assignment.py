from typing import List, Dict
import time
import psutil
import random

class CreditAssignment:
    """Handles component-level credit assignment for ML pipelines."""
    
    def __init__(self):
        self.component_credits = {}  # Stores credit information for each component
        self.component_history = {}  # Add component history for tracking
        self.evaluation_time_history = []  # Track evaluation times
        
    def ablation_credit(self, pipeline: List[str], 
                        base_performance: float, 
                        evaluate_fn) -> Dict[str, float]:
        """
        Assign credit to pipeline components based on ablation analysis.
        
        Args:
            pipeline: List of component names in the pipeline
            base_performance: Performance of the complete pipeline
            evaluate_fn: Function to evaluate modified pipelines
            
        Returns:
            Dictionary mapping component names to credit values
        """
        credits = {}
        
        # For each component, evaluate the pipeline without it
        for i, component in enumerate(pipeline):
            if i == 0 or i == len(pipeline) - 1:
                # Can't remove first or last component, assign default credit
                credits[component] = 0.5
                continue
                
            # Create a new pipeline without this component
            modified_pipeline = pipeline.copy()
            modified_pipeline.pop(i)
            
            # Evaluate the modified pipeline
            mod_performance = evaluate_fn(modified_pipeline)
            
            # Credit is proportional to the performance drop when component is removed
            perf_drop = base_performance - mod_performance
            credits[component] = max(0, perf_drop)  # Ensure non-negative credit
            
        # Normalize credits
        total_credit = sum(credits.values())
        if (total_credit > 0):
            for component in credits:
                credits[component] /= total_credit
                
        return credits
    
    def update_component_credits(self, component_credits: Dict[str, float], 
                                performance: float):
        """
        Update the stored credit values for components.
        
        Args:
            component_credits: Dictionary of component credits from current pipeline
            performance: Overall performance of the pipeline
        """
        for component, credit in component_credits.items():
            if component not in self.component_credits:
                self.component_credits[component] = {
                    "count": 0,
                    "total_credit": 0.0,
                    "avg_credit": 0.0,
                    "weighted_performance": 0.0
                }
                
            info = self.component_credits[component]
            info["count"] += 1
            info["total_credit"] += credit
            info["avg_credit"] = info["total_credit"] / info["count"]
            info["weighted_performance"] += credit * performance
    
    def get_component_value(self, component: str) -> float:
        """
        Get the estimated value of a component based on historical credit.
        
        Args:
            component: The component name
            
        Returns:
            Estimated value of the component
        """
        if component not in self.component_credits:
            return 0.0
            
        info = self.component_credits[component]
        if info["count"] == 0:
            return 0.0
            
        # Value is based on average credit and weighted performance
        return info["avg_credit"] * (info["weighted_performance"] / info["total_credit"])
    
    def assign_marl_credit(self, pipeline, student_actions, teacher_interventions, 
                          performance, base_student_performance=None):
        """
        Assign credit between teacher and student for MARL setting
        
        Args:
            pipeline: The pipeline components
            student_actions: List of actions student would have taken
            teacher_interventions: List of teacher intervention decisions
            performance: Overall performance achieved
            base_student_performance: Estimated performance if student acted alone
            
        Returns:
            Dictionary with credit assigned to each agent
        """
        # Simple differentiation if no base performance provided
        if base_student_performance is None:
            base_student_performance = performance * 0.8  # Assume 80% performance
        
        # Calculate intervention impact
        intervention_count = sum(1 for i in teacher_interventions if i)
        
        if intervention_count == 0:
            # No interventions - student gets all credit
            return {
                'student_credit': 1.0,
                'teacher_credit': 0.0,
                'improvement': 0.0
            }
        
        # Calculate improvement due to teacher
        improvement = performance - base_student_performance
        
        # Calculate credit ratio based on improvement
        if improvement <= 0:
            # No improvement or negative - student gets most credit
            student_credit = 0.9
            teacher_credit = 0.1
        else:
            # Positive improvement - teacher gets credit proportional to improvement
            # Normalize improvement to be between 0 and 1
            norm_improvement = min(1.0, improvement / 0.2)  # 0.2 is a significant improvement
            
            # Teacher credit proportional to normalized improvement
            teacher_credit = 0.2 + 0.6 * norm_improvement  # Between 0.2 and 0.8
            student_credit = 1.0 - teacher_credit
        
        return {
            'student_credit': student_credit,
            'teacher_credit': teacher_credit,
            'improvement': improvement
        }
    
    def assign_component_credit(self, pipeline_components, performance, evaluate_fn):
        """Component-based credit assignment that adapts to available resources"""
        
        # Skip empty pipelines
        if not pipeline_components:
            return {}
            
        # Single component pipelines
        if len(pipeline_components) == 1:
            return {str(pipeline_components[0]): 1.0}
        
        # Check system resources to adapt
        mem = psutil.virtual_memory()
        resource_level = mem.available / mem.total
        
        # Adaptive sampling based on pipeline length and resources
        sample_rate = 1.0
        if len(pipeline_components) > 5:
            sample_rate = min(1.0, resource_level * 1.5)
        
        # Select components to evaluate
        n_samples = max(2, int(len(pipeline_components) * sample_rate))
        components_to_evaluate = random.sample(range(len(pipeline_components) - 1), 
                                              min(n_samples, len(pipeline_components) - 1))
        
        # Track start time for adaptive evaluation
        start_time = time.time()
        component_credits = {}
        
        # Evaluate selected components
        for i in components_to_evaluate:
            component = pipeline_components[i]
            component_str = str(component)
            
            # Skip END_PIPELINE
            if component_str == "END_PIPELINE":
                continue
                
            # Create modified pipeline
            modified = pipeline_components.copy()
            modified.pop(i)
            
            # Evaluate with time constraints
            try:
                modified_performance = evaluate_fn(modified)
                performance_drop = max(0, performance - modified_performance)
                component_credits[component_str] = performance_drop
            except Exception as e:
                print(f"Evaluation error: {e}")
                # Use historical data if available
                if component_str in self.component_history:
                    component_credits[component_str] = self.component_history[component_str].get('avg_value', 0.1)
                else:
                    component_credits[component_str] = 0.1
        
        # Update time history
        eval_time = time.time() - start_time
        self.evaluation_time_history.append(eval_time)
        if len(self.evaluation_time_history) > 10:
            self.evaluation_time_history.pop(0)
        
        # Handle non-evaluated components
        for i, component in enumerate(pipeline_components):
            component_str = str(component)
            if i not in components_to_evaluate and i < len(pipeline_components) - 1 and component_str != "END_PIPELINE":
                # Use historical data if available
                if component_str in self.component_history:
                    component_credits[component_str] = self.component_history[component_str].get('avg_value', 0.1)
                else:
                    avg_credit = sum(component_credits.values()) / max(1, len(component_credits))
                    component_credits[component_str] = avg_credit
        
        # Last component gets small credit
        if pipeline_components and str(pipeline_components[-1]) == "END_PIPELINE":
            component_credits["END_PIPELINE"] = 0.05
        
        # Normalize credits
        total_credit = sum(component_credits.values())
        if total_credit > 0:
            for component in component_credits:
                component_credits[component] /= total_credit
                
                # Update history
                if component not in self.component_history:
                    self.component_history[component] = {
                        'count': 0,
                        'total_value': 0,
                        'avg_value': 0
                    }
                    
                history = self.component_history[component]
                history['count'] += 1
                history['total_value'] += component_credits[component]
                history['avg_value'] = history['total_value'] / history['count']
        
        return component_credits
        
    def translate_component_to_agent_credit(self, component_credits, pipeline_components, action_sources):
        """Convert component credits to agent credits"""
        student_credit = 0
        teacher_credit = 0
        
        for i, component in enumerate(pipeline_components):
            component_str = str(component)
            if component_str in component_credits:
                if i < len(action_sources) and action_sources[i] == "teacher":
                    teacher_credit += component_credits[component_str]
                else:
                    student_credit += component_credits[component_str]
        
        # Normalize
        total = student_credit + teacher_credit
        if total > 0:
            student_credit /= total
            teacher_credit /= total
            
        return {
            'student_credit': student_credit,
            'teacher_credit': teacher_credit
        }
