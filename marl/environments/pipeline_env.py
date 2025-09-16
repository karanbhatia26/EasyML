import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import gym
from gym import spaces
from marl.utils.component_guide import ComponentTransitionRules
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from marl.environments.ml_components import COMPONENT_MAP, COMPONENT_META
import signal
from collections import deque, Counter

class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException

class PipelineEnvironment(gym.Env):
    """Environment for building ML pipelines."""
    
    def __init__(self, dataset, available_components=None, max_pipeline_length=10, debug=False):
        super().__init__()
        
        # Make sure dataset is a dictionary with all required keys
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        for key in required_keys:
            if key not in dataset:
                raise KeyError(f"Dataset missing required key: {key}")
        
        self.X_train = dataset['X_train'] 
        self.y_train = dataset['y_train']
        self.X_val = dataset['X_val']
        self.y_val = dataset['y_val']
        self.X_test = dataset['X_test']
        self.y_test = dataset['y_test']
        
        # Available ML components
        self.available_components = available_components or self._default_components()
        self.component_ids = {comp: i for i, comp in enumerate(self.available_components)}
        
        # Environment parameters
        self.max_pipeline_length = max_pipeline_length
        self.debug = debug
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.available_components))
        
        # Observation space: one-hot encoding of current pipeline components
        # plus dataset characteristics
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(len(self.available_components) * max_pipeline_length + 10,), 
            dtype=np.float32
        )
        
        # Current state
        self.current_pipeline = []
        self.last_performance = 0.0
        # Track component usage for overuse penalties
        self.usage_window = deque(maxlen=200)
        self.usage_counter = Counter()
        
        self.transition_rules = ComponentTransitionRules()
        self.fixed_state_dim = None
        
        # Add stats tracking
        self.stats = {
            'total_pipelines': 0,
            'incompatible_pipelines': 0,
            'incompatible_reasons': {},
            'timeout_pipelines': 0,
            'exception_pipelines': 0,
            'successful_pipelines': 0
        }
        
    def _default_components(self):
        """Define default available components if none provided."""
        components = [
            
            # Imputers for missing values
            "SimpleImputer(strategy='mean')",
            "SimpleImputer(strategy='median')",
            "SimpleImputer(strategy='most_frequent')",
            "SimpleImputer(strategy='constant', fill_value=0)",

            # Scalers
            "StandardScaler()",
            "MinMaxScaler()",
            "RobustScaler()",
            "MaxAbsScaler()",
            "Normalizer()",

            # Encoders
            "OneHotEncoder(drop='first')",
            "OrdinalEncoder()",

            # Feature selection
            "SelectKBest(k=5)",
            "SelectKBest(k=10)",
            "SelectPercentile(percentile=50)",
            "VarianceThreshold(threshold=0.1)",

            # Feature transformation
            "PCA(n_components=2)",
            "PCA(n_components=5)",
            "PCA(n_components=10)",
            "TruncatedSVD(n_components=5)",
            "PolynomialFeatures(degree=2)",
            "Nystroem(kernel='rbf', n_components=100)",
            "RBFSampler(gamma=0.1, n_components=100)",
            "KernelPCA(n_components=50, kernel='rbf')",
            "QuantileTransformer(output_distribution='normal')",
            "PowerTransformer()",
            # Classifiers
            "LogisticRegression(max_iter=1000)",
            "LogisticRegression(max_iter=1000, C=0.1)",
            "LogisticRegression(max_iter=1000, C=10)",
            "DecisionTreeClassifier(max_depth=5)",
            "DecisionTreeClassifier(max_depth=10)",
            "DecisionTreeClassifier(max_depth=None)",
            "RandomForestClassifier(n_estimators=50)",
            "RandomForestClassifier(n_estimators=100)",
            "RandomForestClassifier(n_estimators=200)",
            "GradientBoostingClassifier(n_estimators=100)",
            "Kneighbors(n_neighbors=5)",
            "SVC(kernel='rbf', C=10)",
            "SVC(kernel='poly', degree=3)",
            "LinearSVC(max_iter=2000)",
            "MLP(hidden_layer_sizes=(100,), max_iter=500)",
            "AdaBoost()",
            "HistGradientBoostingClassifier()",

            # End pipeline marker
            "END_PIPELINE"
        ]
        return components
        
    def _get_state_representation(self):
        """Enhanced state representation with pipeline composition indicators"""
        # Create binary flags for component types
        has_imputer = 0
        has_scaler = 0
        has_dim_reducer = 0
        has_encoder = 0
        has_classifier = 0
        
        # Check pipeline components
        for comp in self.current_pipeline:
            if 'SimpleImputer' in comp:
                has_imputer = 1
            elif any(x in comp for x in ["Scaler", "Normalizer", "MinMax", "MaxAbs", "RobustScaler",
                                         "QuantileTransformer", "PowerTransformer"]):
                has_scaler = 1
            elif any(x in comp for x in ['PCA', 'SelectK', 'SelectPercentile', 'Variance', 'TruncatedSVD', 'Nystroem', 'KernelPCA']):
                has_dim_reducer = 1
            elif 'Encoder' in comp:
                has_encoder = 1
            elif any(x in comp for x in ['Classifier', 'Regression', 'KNeighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']):
                has_classifier = 1
        
        # Add these binary flags to state representation
        pipeline_state = [has_imputer, has_scaler, has_dim_reducer, has_encoder, has_classifier]
        
        # Dataset characteristics - handle numeric vs categorical columns properly
        data_chars = []
        
        # 1. Basic dataset properties
        data_chars.append(self.X_train.shape[0] / 10000)  # Normalized sample count
        data_chars.append(self.X_train.shape[1] / 100)    # Normalized feature count
        data_chars.append(len(np.unique(self.y_train)) / 10)  # Normalized class count
        
        # 2. Identify numeric columns and calculate statistics only on those
        numeric_cols = self.X_train.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_data = self.X_train[numeric_cols]
            # Calculate variance on numeric columns only
            try:
                data_chars.append(np.mean(numeric_data.var(numeric_only=True)))
            except:
                data_chars.append(0.0)  # Default if calculation fails
        else:
            data_chars.append(0.0)  # No numeric columns
        
        # 3. Count of categorical vs numeric columns
        total_cols = self.X_train.shape[1]
        categorical_cols = total_cols - len(numeric_cols)
        data_chars.append(categorical_cols / total_cols)  # Ratio of categorical features
        
        # 4. Missing value statistics - handle all column types
        missing_counts = self.X_train.isna().sum()
        if len(missing_counts) > 0:
            missing_ratio = missing_counts.sum() / (self.X_train.shape[0] * total_cols)
            data_chars.append(missing_ratio)
        else:
            data_chars.append(0.0)  # No missing values
        
        # Add some placeholder values to ensure consistent shape
        while len(data_chars) < 10:
            data_chars.append(0.0)
        
        # Ensure we don't exceed 10 features
        data_chars = data_chars[:10]
        
        # Calculate raw state as before
        raw_state = np.concatenate([pipeline_state, np.array(data_chars)])
        
        # NEW: Ensure consistent dimensions throughout episode
        if hasattr(self, 'fixed_state_dim') and self.fixed_state_dim is not None:
            current_dim = len(raw_state)
            if current_dim != self.fixed_state_dim:
                if current_dim < self.fixed_state_dim:
                    # Pad shorter state with zeros
                    raw_state = np.pad(raw_state, (0, self.fixed_state_dim - current_dim))
                else:
                    # Truncate longer state
                    raw_state = raw_state[:self.fixed_state_dim]
        
        return raw_state
        
    def _evaluate_pipeline(self, pipeline=None):
        """Evaluate the current pipeline with better error handling."""
        pipeline_to_evaluate = pipeline if pipeline is not None else self.current_pipeline
        
        if not pipeline_to_evaluate:
            return 0.0
            
        # Remove END_PIPELINE token if present
        pipeline_components = [comp for comp in pipeline_to_evaluate if comp != "END_PIPELINE"]
        
        # # Use the same CLASSIFIER_NAMES as in pipeline_is_incompatible
        # CLASSIFIER_NAMES = [
        #     "Classifier", "Regressor", "SVC", "LogisticRegression", 
        #     "RandomForest", "GradientBoostingClassifier", "KNeighbors", 
        #     "MLP", "AdaBoost", "LinearSVC",
        #     "HistGradientBoostingClassifier"
        # ]
        
        if not pipeline_components or not is_classifier(pipeline_components[-1]):
            print("Pipeline evaluation skipped: Pipeline must end with a classifier")
            return 0.0
        
        try:
            # Create a scikit-learn Pipeline object
            steps = []
            
            # First, detect if we need special column handling
            categorical_cols = self.X_train.select_dtypes(include=['category', 'object']).columns
            numeric_cols = self.X_train.select_dtypes(include=['number']).columns
            
            # Only add column handling if we have mixed column types
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import OneHotEncoder, StandardScaler
                from sklearn.impute import SimpleImputer
                
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ])
                
                steps.append(('preprocessor', preprocessor))
            
            for i, component_str in enumerate(pipeline_components):
                if component_str == "END_PIPELINE":
                    continue
                    
                if component_str not in COMPONENT_MAP:
                    print(f"Pipeline evaluation failed: Component not found: {component_str}")
                    return 0.0
                    
                component = COMPONENT_MAP[component_str]
                steps.append((f'step_{i}', component))
            
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._fit_and_score_pipeline, steps)
                    try:
                        score = future.result(timeout=300)
                        self.stats['total_pipelines'] += 1
                        self.stats['successful_pipelines'] += 1
                        return score
                    except concurrent.futures.TimeoutError:
                        print("Pipeline evaluation timed out after 300 seconds")
                        self.stats['total_pipelines'] += 1
                        self.stats['timeout_pipelines'] += 1
                        return 0.0
            except Exception as inner_e:
                print(f"Pipeline evaluation failed during fitting/scoring: \n{str(inner_e)}")
                self.stats['total_pipelines'] += 1
                self.stats['exception_pipelines'] += 1
                return 0.0
                
        except Exception as e:
            print(f"Pipeline evaluation failed: \n{str(e)}")
            self.stats['total_pipelines'] += 1
            self.stats['exception_pipelines'] += 1
            return 0.0

    def _fit_and_score_pipeline(self, steps):
        """Helper method to fit and score pipeline with proper error handling."""
        try:
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train)
            score = pipeline.score(self.X_val, self.y_val)
            return score
        except Exception as e:
            print(f"Pipeline fitting/scoring error: {str(e)}")
            return 0.0
    
    def evaluate_pipeline_on_test(self, pipeline_components):
        """Evaluate a pipeline on the true test set to get unbiased performance."""
        steps = []
        
        pipeline_components = [comp for comp in pipeline_components if comp != "END_PIPELINE"]
        
        if not pipeline_components or not any(model_name in pipeline_components[-1] 
                for model_name in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                   'GradientBoostingClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']):
            print("Pipeline evaluation skipped: Pipeline must end with a classifier")
            return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}
        
        try:
            for i, component_str in enumerate(pipeline_components):
                if component_str == "END_PIPELINE":
                    continue
                    
                if component_str not in COMPONENT_MAP:
                    print(f"Pipeline evaluation failed: Component not found: {component_str}")
                    return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}
                    
                component = COMPONENT_MAP[component_str]
                steps.append((f'step_{i}', component))
            
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train)
            val_score = pipeline.score(self.X_val, self.y_val)
            test_score = pipeline.score(self.X_test, self.y_test)
            
            return {
                'val_score': val_score,
                'test_score': test_score,
                'gap': val_score - test_score
            }
        except Exception as e:
            print(f"Test evaluation error: {str(e)}")
            return {'val_score': 0.0, 'test_score': 0.0, 'gap': 0.0}

    def compare_with_baselines(self, pipeline_components):
        """Compare your pipeline with baseline models."""
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        pipeline_result = self.evaluate_pipeline_on_test(pipeline_components)
        
        baselines = {}
        
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(self.X_train, self.y_train)
        baselines["Majority Class"] = dummy.score(self.X_test, self.y_test)
        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        baselines["Logistic Regression"] = lr.score(self.X_test, self.y_test)
        
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.X_train, self.y_train)
        # FIX: score must use y_test, not y_train
        baselines["Random Forest"] = rf.score(self.X_test, self.y_test)
        
        results = {
            "Your Pipeline": pipeline_result['test_score'],
            **baselines
        }
        
        print("\nModel Performance Comparison:")
        print("="*40)
        for model, score in results.items():
            print(f"{model:20s}: {score:.4f}")
        
        best_baseline = max(baselines.values())
        improvement = (pipeline_result['test_score'] - best_baseline) / best_baseline * 100 if best_baseline > 0 else 0
        
        return {
            'pipeline_score': pipeline_result['test_score'],
            'best_baseline': best_baseline,
            'improvement': improvement,
            'all_results': results
        }

    def cross_validate_pipeline(self, pipeline_components, cv=5):
        """Test pipeline stability through cross-validation."""
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        pipeline_components = [comp for comp in pipeline_components if comp != "END_PIPELINE"]
        
        steps = []
        for i, component_str in enumerate(pipeline_components):
            if component_str == "END_PIPELINE":
                continue
                
            if component_str not in COMPONENT_MAP:
                print(f"Pipeline evaluation failed: Component not found: {component_str}")
                # Include status so callers can track failures
                return {'mean_score': 0.0, 'std_score': 0.0, 'stability': 0.0, 'all_scores': [], 'status': 'exception'}
                
            component = COMPONENT_MAP[component_str]
            steps.append((f'step_{i}', component))
        
        try:
            pipeline = Pipeline(steps)
            
            import pandas as pd
            X_combined = pd.concat([self.X_train, self.X_val])
            y_combined = np.concatenate([self.y_train, self.y_val])
            
            cv_scores = cross_val_score(pipeline, X_combined, y_combined, cv=cv)
            
            return {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'all_scores': cv_scores,
                'stability': 1 - (cv_scores.std() / cv_scores.mean()) if cv_scores.mean() > 0 else 0,
                'status': 'success'
            }
        except Exception as e:
            print(f"Cross-validation error: {str(e)}")
            return {'mean_score': 0.0, 'std_score': 0.0, 'stability': 0.0, 'all_scores': [], 'status': 'exception'}

    def validate_pipeline_results(self, best_pipeline):
        """Comprehensive validation of pipeline results."""
        print(f"\n{'='*20} PIPELINE VALIDATION {'='*20}\n")
        
        print("\n[1] Test Set Performance:")
        test_results = self.evaluate_pipeline_on_test(best_pipeline)
        print(f"Validation score: {test_results['val_score']:.4f}")
        print(f"Test score: {test_results['test_score']:.4f}")
        print(f"Gap (overfitting indicator): {test_results['gap']:.4f}")
        
        print("\n[2] Baseline Comparison:")
        baseline_results = self.compare_with_baselines(best_pipeline)
        print(f"Improvement over best baseline: {baseline_results['improvement']:.2f}%")
        
        print("\n[3] Stability Analysis (Cross-validation):")
        cv_results = self.cross_validate_pipeline(best_pipeline)
        print(f"Mean CV score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"Stability score: {cv_results['stability']:.4f} (higher is better)")
        
        print("\n[4] Final Assessment:")
        overall_quality = "Poor"
        if test_results['test_score'] > 0.7 and baseline_results['improvement'] > 5 and cv_results['stability'] > 0.9:
            overall_quality = "Excellent"
        elif test_results['test_score'] > 0.6 and baseline_results['improvement'] > 2 and cv_results['stability'] > 0.8:
            overall_quality = "Good"
        elif test_results['test_score'] > 0.5 and baseline_results['improvement'] > 0:
            overall_quality = "Fair"
        
        print(f"Overall pipeline quality: {overall_quality}")
        
        return {
            'test_performance': test_results,
            'baseline_comparison': baseline_results,
            'stability': cv_results,
            'overall_quality': overall_quality
        }
    
    def reset(self):
        """Reset the environment for a new episode with dimension stability."""
        self.current_pipeline = []
        self.last_performance = 0.0
        
        # Get initial state representation
        state = self._get_state_representation()
        
        # Store the initial dimension to maintain consistency throughout the episode
        if self.fixed_state_dim is None:
            self.fixed_state_dim = len(state)
            print(f"Environment initialized with fixed state dimension: {self.fixed_state_dim}")
        
        return state
    
    def step(self, action):
        """Take an action and return the next state, reward, done flag, and info"""
        # Check if it's a valid action to prevent redundancy
        valid_actions = self.get_filtered_actions()
        if action not in valid_actions:
            print(f"Warning: Action {action} ({self.available_components[action]}) is invalid/redundant but was attempted")
        
        component = self.available_components[action]
        self.current_pipeline.append(str(component))
        # usage tracking
        self.usage_window.append(str(component))
        self.usage_counter[str(component)] += 1
        
        # Check for incompatibilities
        incompatibility = pipeline_is_incompatible(self.current_pipeline)
        if incompatibility:
            self.stats['total_pipelines'] += 1
            self.stats['incompatible_pipelines'] += 1
            
            # Track reason
            reason = str(incompatibility).split(': ')[1] if ': ' in str(incompatibility) else str(incompatibility)
            if reason not in self.stats['incompatible_reasons']:
                self.stats['incompatible_reasons'][reason] = 0
            self.stats['incompatible_reasons'][reason] += 1
            
            if self.debug:
                print(incompatibility)
            # Reduced negative reward for incompatible pipelines
            incompatible_reward = -0.3  # slightly stronger discourage
            # Remove the incompatible component
            removed_component = self.current_pipeline.pop()
            if self.debug:
                print(f"Removed incompatible component: {removed_component}")
            next_state = self._get_state_representation()
            return next_state, incompatible_reward, False, {"performance": 0.0}
        
        # Check for redundant components
        redundancy_penalty = 0.0
        if len(set(self.current_pipeline)) != len(self.current_pipeline):
            redundancy_penalty = -0.25
            if self.debug:
                print(f"Redundant: Pipeline contains duplicate components: {redundancy_penalty}")
        
        done = False
        if str(component) == "END_PIPELINE" or len(self.current_pipeline) >= self.max_pipeline_length:
            done = True
        
        next_state = self._get_state_representation()
        
        if done:
            # Prefer CV for stability; capture status
            cv_stats = self.cross_validate_pipeline([str(c) for c in self.current_pipeline], cv=3)
            performance = cv_stats.get('mean_score', 0.0)
            eval_status = cv_stats.get('status', 'success')
            
            # If CV failed or produced non-finite/zero performance, fallback to direct eval with timeout
            if eval_status != 'success' or not np.isfinite(performance) or performance <= 0.0:
                eval_score, eval_status = self.evaluate_with_timeout(self.current_pipeline, timeout=120, return_status=True)
                performance = eval_score
            
            # Base reward is performance
            reward = performance
            
            # Apply penalties and bonuses
            pipeline_length = len(self.current_pipeline)
            
            # Penalize very short pipelines more strongly
            if pipeline_length <= 1:
                short_penalty = -0.3
                if self.debug:
                    print(f"Very short pipeline penalty: {short_penalty}")
                reward += short_penalty
            elif pipeline_length <= 2:
                short_penalty = -0.2
                if self.debug:
                    print(f"Short pipeline penalty: {short_penalty}")
                reward += short_penalty
            
            # Analyze component types in the pipeline
            components_str = " ".join(self.current_pipeline)
            
            # Reward for balanced preprocessing
            has_imputer = any('Imputer' in c for c in self.current_pipeline)
            has_scaler = any(x in components_str for x in ["Scaler", "Normalizer", "MinMax", "MaxAbs", "RobustScaler", "QuantileTransformer", "PowerTransformer"])
            has_feature_selection = any(x in components_str for x in ['SelectK', 'SelectPercentile', 'Variance'])
            
            # Reward proper preprocessing steps
            preprocessing_bonus = 0.1
            if has_imputer and has_scaler and has_feature_selection:
                preprocessing_bonus = 0.3
                if self.debug:
                    print(f"Complete preprocessing bonus: {preprocessing_bonus}")
            elif (has_imputer and has_scaler) or (has_scaler and has_feature_selection) or (has_imputer and has_feature_selection):
                preprocessing_bonus = 0.2
                if self.debug:
                    print(f"Partial preprocessing bonus: {preprocessing_bonus}")
            elif has_imputer or has_scaler or has_feature_selection:
                preprocessing_bonus = 0.15
                if self.debug:
                    print(f"Basic preprocessing bonus: {preprocessing_bonus}")
            
            reward += preprocessing_bonus
            
            # Apply redundancy penalty
            reward += redundancy_penalty
            
            # Exploration bonus - reward for trying new component combinations
            if not hasattr(self, 'seen_pipelines'):
                self.seen_pipelines = set()
            
            pipeline_signature = tuple(self.current_pipeline)
            if pipeline_signature not in self.seen_pipelines:
                exploration_bonus = 0.05
                if self.debug:
                    print(f"Novel pipeline exploration bonus: {exploration_bonus}")
                reward += exploration_bonus
                self.seen_pipelines.add(pipeline_signature)
            
            # Overuse penalty to reduce fixation on a single component (e.g., PowerTransformer)
            total_use = max(1, len(self.usage_window))
            power_freq = self.usage_window.count("PowerTransformer()")
            freq = power_freq / total_use
            if freq > 0.2:
                overuse_penalty = 0.1 * (freq - 0.2)
                reward -= overuse_penalty
                if self.debug:
                    print(f"Overuse penalty applied: -{overuse_penalty:.3f}")

            # Soft length penalty to discourage overly long pipelines
            length_penalty = 0.02 * pipeline_length
            reward -= length_penalty
            if self.debug and length_penalty:
                print(f"Length penalty: -{length_penalty:.3f}")
            
            # NEW: Update stats counters regardless of evaluation path, and only count genuine successes
            try:
                pipeline_components = [c for c in self.current_pipeline if c != "END_PIPELINE"]
                has_estimator = len(pipeline_components) > 0 and is_classifier(pipeline_components[-1])
            except Exception:
                has_estimator = False
            
            self.stats['total_pipelines'] += 1
            # Count status-based outcomes
            if eval_status == 'timeout':
                self.stats['timeout_pipelines'] += 1
            elif eval_status == 'exception':
                self.stats['exception_pipelines'] += 1
            elif has_estimator and np.isfinite(performance) and performance > 0.0:
                self.stats['successful_pipelines'] += 1
            # else: unsuccessful but not exception/timeout — do not count as successful
            
            return next_state, reward, done, {"performance": performance}
        else:
            # Non-terminal step rewards - REDUCE PENALTIES + role-based redundancy discouragement
            step_reward = -0.005
            
            # Add immediate feedback based on component selection
            component_str = str(component)
            immediate_reward = 0.0
            
            # Reward logical ordering of components
            pipeline_len = len(self.current_pipeline)
            if pipeline_len == 1:
                # First component selection
                if "Imputer" in component_str:
                    immediate_reward = 0.05
                    if self.debug:
                        print(f"Good first component (Imputer): +{immediate_reward}")
                elif any(x in component_str for x in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                    'GradientBoostingClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']):
                    immediate_reward = -0.005
                    if self.debug:
                        print(f"Suboptimal first component (Classifier): {immediate_reward}")
            
            # Add to step() function for non-terminal steps
            if "Imputer" in component_str and len(self.current_pipeline) == 1:
                if self.debug:
                    print("Good first component (Imputer): +0.05")
                immediate_reward += 0.05
            
            # Role-based redundancy penalty: discourage adding another non-repeatable role
            role = COMPONENT_META.get(component_str, {}).get('role')
            if role and not COMPONENT_META.get(component_str, {}).get('repeatable', False):
                if any(COMPONENT_META.get(c, {}).get('role') == role for c in self.current_pipeline[:-1]):
                    step_reward -= 0.02
            
            # Final non-terminal step reward
            reward = step_reward + immediate_reward + redundancy_penalty
            
            return next_state, reward, done, {"performance": 0.0}
    
    def get_valid_actions(self):
        """Get list of valid actions from current state"""
        valid_actions = []
        
        if any(comp == "END_PIPELINE" for comp in self.current_pipeline):
            return valid_actions
        
        has_classifier = any(model_name in ' '.join(self.current_pipeline) 
                   for model_name in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                   'GradientBoostingClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC'])
        
        for i, component in enumerate(self.available_components):
            if has_classifier and component != "END_PIPELINE":
                continue
                
            if self._is_compatible_with_pipeline(str(component)):
                valid_actions.append(i)
        
        return valid_actions

    def get_filtered_actions(self):
        """Get filtered actions using role grammar and incompatibility oracle"""
        valid_actions = self.get_valid_actions()
        filtered_actions = []

        pipeline_components = [str(comp) for comp in self.current_pipeline]
        has_estimator = any(COMPONENT_META.get(c, {}).get('role') == 'estimator' for c in pipeline_components)

        # If an estimator is already present, only allow terminator
        if has_estimator:
            for idx in valid_actions:
                if str(self.available_components[idx]) == "END_PIPELINE":
                    return [idx]
            return []

        for action in valid_actions:
            comp_str = str(self.available_components[action])

            # Skip exact duplicate component
            if comp_str in pipeline_components:
                continue

            # Enforce non-repeatable roles via metadata
            role = COMPONENT_META.get(comp_str, {}).get('role')
            repeatable = COMPONENT_META.get(comp_str, {}).get('repeatable', False)
            if role and not repeatable:
                if any(COMPONENT_META.get(c, {}).get('role') == role for c in pipeline_components):
                    continue

            # Simulate and reject if incompatible
            tmp_pipeline = pipeline_components + [comp_str]
            if pipeline_is_incompatible(tmp_pipeline):
                continue

            filtered_actions.append(action)

        if self.debug:
            print(f"Filtered actions: {filtered_actions}")
        return filtered_actions

    def _is_compatible_with_pipeline(self, component_name):
        if not self.current_pipeline:
            return True
            
        # Don't check "pipeline lacks classifier" for intermediate components
        # Only check for classifier presence if adding END_PIPELINE
        if component_name == 'END_PIPELINE':
            has_classifier = any(is_classifier(comp) for comp in self.current_pipeline)
            if not has_classifier:
                print(f"Incompatible: Cannot end pipeline without a classifier")
                return False
                
        # Check if previous component is a classifier (can't add after a classifier)
        prev_is_classifier = any(clf in self.current_pipeline[-1] 
                          for clf in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier',
                                     'GradientBoostingClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC'])
        if prev_is_classifier and component_name != 'END_PIPELINE':
            print(f"Incompatible: Cannot add {component_name} after a classifier")
            return False

        # Rest of your compatibility checks remain the same
        feature_extractors = ['PCA', 'TruncatedSVD', 'SelectKBest', 'SelectPercentile', 
                              'VarianceThreshold', 'KernelPCA', 'Nystroem', 'RBFSampler']
        encoders = ['OneHotEncoder', 'OrdinalEncoder']
        scalers = ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 
                   'RobustScaler', 'Normalizer', 'QuantileTransformer', 'PowerTransformer']
        imputers = ['SimpleImputer']
        classifiers = ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'Kneighbors', 'GradientBoostingClassifier', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']
        
        has_dim_reduction = any(extractor in ' '.join(self.current_pipeline) for extractor in feature_extractors)
        
        prev_is_classifier = any(clf in self.current_pipeline[-1] for clf in classifiers)
        
        if prev_is_classifier and component_name != 'END_PIPELINE':
            print(f"Incompatible: Cannot add {component_name} after a classifier")
            return False

        # Memory explosion check for polynomials and encoders
        if (('PolynomialFeatures' in component_name and 
             any('OneHotEncoder' in comp for comp in self.current_pipeline)) or
            ('OneHotEncoder' in component_name and 
             any('PolynomialFeatures' in comp for comp in self.current_pipeline))):
            print(f"Incompatible: OneHotEncoder + PolynomialFeatures causes memory explosion")
            return False

        # Check estimated memory for this pipeline
        test_pipeline = self.current_pipeline + [component_name]
        memory_estimate = self.estimate_memory_requirement(test_pipeline)

        # Higher threshold for image datasets
        memory_threshold = 15.0
        if self.X_train.shape[1] > 60:  # Image dataset
            memory_threshold = 15.0
        else:
            memory_threshold = 15.0

        if memory_estimate > memory_threshold:
            print(f"Memory limit exceeded: Pipeline would require ~{memory_estimate:.1f} GB")
            return False

        component_type = None
        for cat in [imputers, scalers, encoders, feature_extractors]:
            if any(c in component_name for c in cat):
                component_type = cat
                break
                
        if component_type:
            existing_same_type = [c for c in self.current_pipeline 
                                 if any(t in c for t in component_type)]
            if len(existing_same_type) >= 1:
                print(f"Redundant: Pipeline already contains {existing_same_type}")
                return False
                
        if any(encoder in component_name for encoder in encoders) and has_dim_reduction:
            print(f"Incompatible: Cannot apply {component_name} after dimension reduction")
            return False
            
        if 'n_components=' in component_name:
            # Extract just the n_components value, handling additional parameters
            n_components_part = component_name.split('n_components=')[1]
            # Get value before comma or closing parenthesis
            n_components_str = n_components_part.split(',')[0].split(')')[0].strip()
            n_components = int(n_components_str)
            max_features = min(self.X_train.shape[0], self.X_train.shape[1])
            if n_components > max_features:
                print(f"Invalid: PCA n_components={n_components} exceeds max of {max_features}")
                return False

        # Enhanced redundancy check - count component types
        imputer_count = sum(1 for c in self.current_pipeline if 'SimpleImputer' in c)
        scaler_count = sum(1 for c in self.current_pipeline 
                      if any(x in c for x in ['Scaler', 'Normalizer', 'MinMax', 'MaxAbs', 'Robust']))
        selector_count = sum(1 for c in self.current_pipeline 
                        if any(x in c for x in ['Select', 'PCA', 'SVD']))
    
        # Limit component repetition
        if 'SimpleImputer' in component_name and imputer_count >= 1:
            print(f"Redundant: Pipeline already has {imputer_count} imputers")
            return False
        if any(x in component_name for x in ['Scaler', 'Normalizer', 'MinMax', 'MaxAbs', 'Robust']) and scaler_count >= 1:
            print(f"Redundant: Pipeline already has {scaler_count} scalers")
            return False
        if any(x in component_name for x in ['Select', 'PCA', 'SVD']) and selector_count >= 1:
            print(f"Redundant: Pipeline already has {selector_count} feature selectors")
            return False

        return True

    def get_teacher_state(self, student_history=None):
        """
        Create state representation for teacher agent
        
        Args:
            student_history: Recent student actions (optional)
            
        Returns:
            State vector for teacher
        """
        base_state = self._get_state_representation()
        
        if student_history:
            student_state = np.zeros(len(self.available_components))
            for action in student_history[-5:]:
                if 0 <= action < len(student_state):
                    student_state[action] += 1
            
            return np.concatenate([base_state, student_state])
        
        return base_state

    def process_teacher_intervention(self, student_action, should_intervene, teacher_action):
        """
        Process teacher's intervention decision
        
        Args:
            student_action: Action selected by student
            should_intervene: Whether teacher wants to intervene
            teacher_action: Teacher's suggested action
            
        Returns:
            Action to execute, source of action
        """
        if should_intervene:
            return teacher_action, "teacher"
        else:
            return student_action, "student"
            
    def calculate_teacher_reward(self, student_action, should_intervene, teacher_action, performance):
        """More balanced teacher reward function"""
        # Non-intervention baseline reward
        if not should_intervene:
            return 0.01
        
        # Compare actions to see if teacher made better choice
        if performance > 0:  # Successful intervention
            return performance * 0.2  # Share of performance
        elif student_action == teacher_action:
            # Teacher agreed with student but intervention was unnecessary
            return -0.05
        else:
            # Teacher intervention that didn't improve or worsen situation
            return -0.01  # Smaller penalty

    def _evaluate_pipeline_performance(self, pipeline):
        """Evaluate pipeline with Windows-compatible timeout protection"""
        import threading
        import time
        
        result = [0.0]
        error_occurred = [False]
        
        def evaluation_thread():
            try:
                # Skip pipeline checks for empty pipelines or non-classifier endings
                if not pipeline or pipeline[-1] == "END_PIPELINE" and len(pipeline) < 2:
                    result[0] = 0.0
                    return
                
                # CHECK FOR MEMORY EXPLOSION
                if not self.is_valid_pipeline(pipeline):
                    print("Pipeline evaluation skipped: Memory requirements too high")
                    result[0] = 0.0
                    return
                
                pipeline_components = [comp for comp in pipeline if comp != "END_PIPELINE"]
                
                if not pipeline_components or not any(model_name in pipeline_components[-1] 
                        for model_name in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                   'GradientBoostingClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']):
                    print("Pipeline evaluation skipped: Pipeline must end with a classifier")
                    result[0] = 0.0
                    return
                
                steps = []
                for i, component_str in enumerate(pipeline_components):
                    if component_str == "END_PIPELINE":
                        continue
                        
                    if component_str not in COMPONENT_MAP:
                        print(f"Pipeline evaluation failed: Component not found: {component_str}")
                        result[0] = 0.0
                        return
                        
                    component = COMPONENT_MAP[component_str]
                    steps.append((f'step_{i}', component))
                
                pipeline_obj = Pipeline(steps)
                pipeline_obj.fit(self.X_train, self.y_train)
                result[0] = pipeline_obj.score(self.X_val, self.y_val)
            
            except Exception as e:
                print(f"Pipeline evaluation error: {str(e)}")
                error_occurred[0] = True
                result[0] = 0.0
        
        thread = threading.Thread(target=evaluation_thread)
        thread.daemon = True
        thread.start()
        
        timeout = 300
        start_time = time.time()
        while thread.is_alive() and time.time() - start_time < timeout:
            thread.join(0.5)
        
        if thread.is_alive():
            print(f"Pipeline evaluation timed out after {timeout} seconds")
            
            if len(pipeline) >= 2:
                bad_pair = (str(pipeline[-2]), str(pipeline[-1]))
                
                if not hasattr(self, 'bad_combinations'):
                    self.bad_combinations = set()
                self.bad_combinations.add(bad_pair)
                print(f"Added bad combination to avoid list: {bad_pair}")
            
            return 0.0
        
        return result[0]

    def evaluate_with_timeout(self, pipeline, timeout=120, return_status=False):
        """Evaluate pipeline with proper timeout handling. If return_status=True, returns (score, status)."""
        import threading
        import time
        
        result = [0.0]
        error_occurred = [False]
        status = ['success']
        
        def evaluation_thread():
            try:
                # Remove END_PIPELINE token if present
                pipeline_components = [comp for comp in pipeline if comp != "END_PIPELINE"]
                
                # Only check for classifier requirement at the END of pipeline construction
                if "END_PIPELINE" in pipeline:
                    if not pipeline_components or not is_classifier(pipeline_components[-1]):
                        print("Pipeline evaluation skipped: Final pipeline must end with a classifier")
                        result[0] = 0.0
                        status[0] = 'exception'
                        return
                        
                from sklearn.base import clone
                from sklearn.pipeline import Pipeline as SklearnPipeline
                    
                steps = []
                for i, component_str in enumerate(pipeline_components):
                    if component_str == "END_PIPELINE":
                        continue
                            
                    if component_str not in COMPONENT_MAP:
                        print(f"Pipeline evaluation failed: Component not found: {component_str}")
                        result[0] = 0.0
                        status[0] = 'exception'
                        return
                        
                    component = clone(COMPONENT_MAP[component_str])
                    steps.append((f'step_{i}', component))
                    
                try:
                    pipeline_obj = SklearnPipeline(steps)
                    pipeline_obj.fit(self.X_train, self.y_train)
                    result[0] = pipeline_obj.score(self.X_val, self.y_val)
                except Exception as inner_e:
                    print(f"Pipeline fitting error: {str(inner_e)}")
                    result[0] = 0.0
                    status[0] = 'exception'
                    
            except Exception as e:
                print(f"Evaluation thread error: {str(e)}")
                error_occurred[0] = True
                result[0] = 0.0
                status[0] = 'exception'
        
        thread = threading.Thread(target=evaluation_thread)
        thread.daemon = True
        thread.start()
        
        start_time = time.time()
        while thread.is_alive() and time.time() - start_time < timeout:
            thread.join(0.5)  # Allow interruption every 0.5 seconds
        
        if thread.is_alive():
            print(f"Pipeline evaluation timed out after {timeout} seconds")
            return 0.0
        
        if return_status:
            return result[0], status[0]
        return result[0]
    
    def _update_pipeline_memory(self, pipeline, performance):
        """Store successful pipelines with proper preprocessing value"""
        if not hasattr(self, 'pipeline_memory'):
            self.pipeline_memory = []
        
        # Calculate pipeline complexity and preprocessing score
        def get_pipeline_value(pl, perf):
            components = [c for c in pl if c != "END_PIPELINE"]
            
            # Count unique component types
            has_imputer = any('Imputer' in c for c in components)
            has_scaler = any(any(x in c for x in ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer', 'PowerTransformer']) for c in components)
            has_feature_sel = any(any(x in c for x in ['SelectK', 'SelectPercentile']) for c in components)
            has_dim_red = any(any(x in c for x in ['PCA', 'TruncatedSVD', 'VarianceThreshold', 'KernelPCA', 'Nystroem', 'RBFSampler']) for c in components)
            has_clf = any(any(x in c for x in ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 
                                               'RandomForestClassifier', 'Kneighbors', 'GradientBoostingClassifier', 'SVC', 
                                               'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']) for c in components)
            
            # Give higher value to pipelines with preprocessing
            preprocessing_value = (has_imputer * 0.05 + has_scaler * 0.05 + 
                                  has_feature_sel * 0.03 + has_dim_red * 0.03)
                                  
            # Adjust performance to value preprocessing
            adjusted_perf = perf * (1.0 + preprocessing_value)
            return adjusted_perf
        
        # Add to memory with adjusted value
        pipeline_entry = {
            'components': pipeline.copy(),
            'performance': performance,
            'adjusted_value': get_pipeline_value(pipeline, performance)
        }
        
        # Add to memory
        self.pipeline_memory.append(pipeline_entry)
        
        # Sort pipeline memory by ADJUSTED value to favor preprocessing
        self.pipeline_memory = sorted(
            self.pipeline_memory, 
            key=lambda x: x['adjusted_value'],
            reverse=True
        )[:5]  # Keep top 5
        
        print("Pipeline memory updated:")
        for i, entry in enumerate(self.pipeline_memory, 1):
            components = [c for c in entry['components'] if c != "END_PIPELINE"]
            print(f"  #{i}: {components} - {entry['performance']:.4f}")

    def evaluate_pipeline(self, pipeline=None):
        """Alias for evaluate_with_timeout that matches your call sites"""
        pipeline_to_evaluate = pipeline if pipeline is not None else self.current_pipeline
        return self.evaluate_with_timeout(pipeline_to_evaluate)

    def estimate_memory_requirement(self, pipeline_components):
        """Estimate memory requirements for a pipeline before execution"""
        if not pipeline_components:
            return 0
            
        # Get dataset info
        n_samples = self.X_train.shape[0]
        n_features = self.X_train.shape[1]
        bytes_per_value = 8  # float64
        
        # For MNIST and similar datasets, use special handling
        is_image_dataset = n_features > 60  # Simple heuristic for image data
        
        # Special handling for image datasets
        if is_image_dataset:
            # Check for specific components that work well with image data
            has_pca = any('PCA' in str(comp) for comp in pipeline_components)
            has_select = any('Select' in str(comp) for comp in pipeline_components)
            
            if has_pca or has_select:
                # These are efficient with image data
                return 2.0  # Reasonable memory estimate
            
            # Check for polynomial features which does cause memory issues
            if any('PolynomialFeatures' in str(comp) for comp in pipeline_components):
                # This is a reasonable limit for MNIST with polynomial features
                return 10.0  # Flag as high but not impossible
        
        # For other dataset types, use the original calculation
        # Track feature growth
        estimated_features = n_features
        
        # Standard calculation for typical datasets
        # [Rest of your original estimation logic]
        
        # Apply a more reasonable cap for all estimations
        estimated_memory_gb = (n_samples * estimated_features * bytes_per_value) / (1024**3)
        
        # Apply a cap based on dataset type
        if is_image_dataset:
            return min(estimated_memory_gb, 8.0)  # Cap at 8GB for image datasets
        
        return min(estimated_memory_gb, 16.0)  # Cap at 16GB for other datasets

    def is_valid_pipeline(self, pipeline):
        if not pipeline:
            return True
            
        memory_estimate = self.estimate_memory_requirement(pipeline)
        if memory_estimate > 10.0:
            print(f"Pipeline memory requirement too high: ~{memory_estimate:.1f} GB")
            return False
        
        # Check for dangerous combinations
        has_poly = any('PolynomialFeatures' in str(comp) for comp in pipeline)
        has_onehot = any('OneHotEncoder' in str(comp) for comp in pipeline)
        if has_poly and has_onehot:
            print("Pipeline contains dangerous combination: PolynomialFeatures + OneHotEncoder")
            return False
        
        return True

    def get_pipeline_statistics(self):
        """Get statistics about pipeline evaluation"""
        success_rate = self.stats['successful_pipelines'] / max(1, self.stats['total_pipelines'])
        
        return {
            'total_pipelines': self.stats['total_pipelines'],
            'successful_pipelines': self.stats['successful_pipelines'],
            'incompatible_pipelines': self.stats['incompatible_pipelines'],
            'timeout_pipelines': self.stats['timeout_pipelines'], 
            'exception_pipelines': self.stats['exception_pipelines'],
            'success_rate': success_rate,
            'incompatible_reasons': self.stats['incompatible_reasons']
        }

    def print_pipeline_statistics(self):
        """Print statistics about pipeline evaluation"""
        stats = self.get_pipeline_statistics()
        print("\n=== Pipeline Statistics ===")
        print(f"Total pipelines: {stats['total_pipelines']}")
        print(f"Successful pipelines: {stats['successful_pipelines']} ({stats['success_rate']:.2%})")
        print(f"Incompatible pipelines: {stats['incompatible_pipelines']} ({stats['incompatible_pipelines']/max(1,stats['total_pipelines']):.2%})")
        print(f"Timeout pipelines: {stats['timeout_pipelines']}")
        print(f"Exception pipelines: {stats['exception_pipelines']}")
        
        print("\nIncompatibility reasons:")
        for reason, count in sorted(stats['incompatible_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")

def is_transformer(component_name):
    classifier_names = ['Classifier', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 
                                   'GradientBoostingClassifierClassifier', 'Kneighbors', 'SVC', 'HistGradientBoostingClassifier', 'MLP', 'AdaBoost', 'LinearSVC']
    return not any(clf in component_name for clf in classifier_names)

def is_classifier(component_name):
    CLASSIFIER_NAMES = [
        "Classifier", "Regressor", "SVC", "LogisticRegression", 
        "RandomForest", "GradientBoostingClassifier", "KNeighbors", 
        "MLP", "AdaBoost", "LinearSVC",
        "HistGradientBoostingClassifier"
    ]
    return any(clf in component_name for clf in CLASSIFIER_NAMES)

def pipeline_is_incompatible(pipeline):
    if not pipeline:
        return False
        
    classifiers = []
    reducers = []
    scalers = []
    encoders = []
    imputers = []
    
    # Updated classifier detection
    CLASSIFIER_NAMES = [
        "Classifier", "Regressor", "SVC", "LogisticRegression", 
        "RandomForest", "GradientBoostingClassifier", "KNeighbors", 
        "MLP", "AdaBoost", "LinearSVC",
        "HistGradientBoostingClassifier"
    ]
    
    # Updated reducer detection
    REDUCER_NAMES = [
        "PCA", "TruncatedSVD", "SelectKBest", "SelectPercentile", 
        "VarianceThreshold", "KernelPCA", "Nystroem", "RBFSampler"]
    
    # Updated encoder detection
    ENCODER_NAMES = ["Encoder", "OneHot", "Label", "Ordinal"]
    
    # Updated scaler detection
    SCALER_NAMES = [
        "Scaler", "Normalizer", "MinMax", "MaxAbs", "RobustScaler",
        "QuantileTransformer", "PowerTransformer"
    ]
    
    for i, component in enumerate(pipeline):
        component_str = str(component)
        
        if any(c in component_str for c in CLASSIFIER_NAMES):
            classifiers.append((i, component_str))
        if any(c in component_str for c in REDUCER_NAMES):
            reducers.append((i, component_str))
        if any(c in component_str for c in SCALER_NAMES):
            scalers.append((i, component_str))
        if any(c in component_str for c in ENCODER_NAMES):
            encoders.append((i, component_str))
        if "Imputer" in component_str:
            imputers.append((i, component_str))
    
    for encoder_idx, _ in encoders:
        if any(reducer_idx < encoder_idx for reducer_idx, _ in reducers):
            return f"Incompatible: Cannot apply encoding after dimension reduction"
    
    if len(classifiers) > 1:
        if classifiers[-1][0] != len(pipeline) - 1:
            return f"Incompatible: Multiple classifiers in pipeline"
    
    if len(reducers) > 1:
        consecutive = all(reducers[i+1][0] - reducers[i][0] == 1 for i in range(len(reducers)-1))
        if not consecutive:
            return f"Incompatible: Non-sequential dimension reduction"
    
    for scaler_idx, _ in scalers:
        if any(reducer_idx < scaler_idx for reducer_idx, _ in reducers):
            return f"Incompatible: Scaling should be applied before dimension reduction"
    
    for classifier_idx, _ in classifiers[:-1]:
        if any(component_idx > classifier_idx for component_type in [scalers, encoders, imputers, reducers] 
               for component_idx, _ in component_type):
            return f"Incompatible: Preprocessing after classifier"
    
    for i in range(len(pipeline) - 1):
        if str(pipeline[i]) == str(pipeline[i + 1]):
            return f"Redundant: Pipeline contains consecutive duplicate components"
    
    for i, component in enumerate(pipeline):
        if any(s in str(component) for s in ["SelectKBest", "SelectPercentile"]):
            if any("PCA" in str(c) or "TruncatedSVD" in str(c) for c in pipeline[:i]):
                return f"Incompatible: Feature selection after dimension reduction"
    
    for imputer_idx, _ in imputers:
        if any(encoder_idx < imputer_idx for encoder_idx, _ in encoders):
            return f"Incompatible: Imputation should be done before encoding"
    
    if len(scalers) > 1:
        scaler_names = [name for _, name in scalers]
        if any("StandardScaler" in s for s in scaler_names) and any("MinMaxScaler" in s for s in scaler_names):
            return f"Incompatible: Using both StandardScaler and MinMaxScaler"
    if "END_PIPELINE" in str(pipeline[-1]):
        # Check if we have any classifier in the pipeline
        has_classifier = any(is_classifier(component) for component in pipeline[:-1])  # Exclude END_PIPELINE
        
        if not has_classifier:
            return f"Incompatible: Final pipeline must end with a classifier"
        
    return False