import numpy as np
import pandas as pd

class ComponentGuide:
    def __init__(self):
        self.transition_scores = {}
        self.component_performances = {}
        
    def update(self, pipeline, performance):
        if len(pipeline) < 2:
            return
            
        # Update component individual performances
        for component in pipeline:
            if component not in self.component_performances:
                self.component_performances[component] = []
            self.component_performances[component].append(performance)
        
        # Update transition pairs
        for i in range(len(pipeline) - 1):
            from_comp = pipeline[i]
            to_comp = pipeline[i+1]
            
            if from_comp not in self.transition_scores:
                self.transition_scores[from_comp] = {}
                
            if to_comp not in self.transition_scores[from_comp]:
                self.transition_scores[from_comp][to_comp] = []
                
            self.transition_scores[from_comp][to_comp].append(performance)
    
    def get_transition_preference(self, current_component):
        if current_component not in self.transition_scores:
            return {}
            
        preferences = {}
        for next_comp, scores in self.transition_scores[current_component].items():
            preferences[next_comp] = np.mean(scores)
            
        return preferences
    
    def get_component_performance(self, component):
        if component not in self.component_performances:
            return 0.0
        return np.mean(self.component_performances[component])
        
    def suggest_next_component(self, current_component, available_components):
        prefs = self.get_transition_preference(current_component)
        
        # If no transitions recorded, return None
        if not prefs:
            return None
            
        # Filter to only available components
        available_prefs = {comp: score for comp, score in prefs.items() 
                          if comp in available_components}
        
        if not available_prefs:
            return None
            
        # Return the highest performing next component
        return max(available_prefs.items(), key=lambda x: x[1])[0]

class ComponentTransitionRules:
    def __init__(self):
        self.preprocessors = ["SimpleImputer", "StandardScaler", "MinMaxScaler", 
                             "RobustScaler", "MaxAbsScaler", "Normalizer"]
        
        self.feature_transformers = ["PCA", "SelectKBest", "SelectPercentile", 
                                    "VarianceThreshold", "TruncatedSVD", "PolynomialFeatures"]
        
        self.encoders = ["OneHotEncoder", "OrdinalEncoder"]
        
        self.classifiers = ["LogisticRegression", "DecisionTreeClassifier", 
                           "RandomForestClassifier", "GradientBoostingClassifier", 
                            "KNeighborsClassifier"]
        
        # Define allowed transitions between component groups
        self.allowed_transitions = {
            "START": self.preprocessors + self.feature_transformers + self.encoders,
            
            # Preprocessor transitions
            **{p: self.preprocessors + self.feature_transformers + self.encoders + self.classifiers 
               for p in self.preprocessors},
            
            # Feature transformer transitions
            **{t: self.preprocessors + self.feature_transformers + self.classifiers 
               for t in self.feature_transformers},
            
            # Encoder transitions
            **{e: self.preprocessors + self.feature_transformers + self.classifiers 
               for e in self.encoders},
            
            # Classifiers (terminal components)
            **{c: [] for c in self.classifiers}
        }
    
    def get_valid_next_components(self, current_pipeline):
        if not current_pipeline:
            current_type = "START"
        else:
            # Extract component type from the current component
            current = current_pipeline[-1]
            current_type = next((comp_type for comp_type in self.allowed_transitions.keys() 
                               if comp_type in current), None)
                               
        if current_type:
            return self.allowed_transitions[current_type]
        return []
    
    def filter_valid_actions(self, current_pipeline, available_components, action_indices):
        if not current_pipeline:
            # For empty pipeline, all preprocessors and feature transformers are valid
            return [i for i in action_indices 
                   if any(p in available_components[i] for p in 
                         self.preprocessors + self.feature_transformers)]
        
        # Get valid next component types
        valid_next = self.get_valid_next_components(current_pipeline)
        
        # Filter actions
        return [i for i in action_indices 
               if any(comp_type in available_components[i] for comp_type in valid_next)]