from typing import List, Dict
class CreditAssignment:
    """Handles component-level credit assignment for ML pipelines."""
    
    def __init__(self):
        self.component_credits = {}  # Stores credit information for each component
        
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
