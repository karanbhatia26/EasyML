# Import all required ML components
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    MaxAbsScaler, Normalizer, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, VarianceThreshold, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures

# ADD GPU COMPONENTS HERE - START
import torch

# Check if GPU is available
GPU_AVAILABLE = torch.cuda.is_available()
print(f"GPU acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")

# Only import cuML if GPU is available
if GPU_AVAILABLE:
    try:
        # Import GPU-accelerated ML components
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler
        from cuml.linear_model import LogisticRegression as cuLogisticRegression
        from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
        from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
        from cuml.decomposition import PCA as cuPCA
        from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
        from cuml.preprocessing import SimpleImputer as cuSimpleImputer
        print("Successfully imported cuML GPU components")
    except ImportError:
        print("cuML not available. Using CPU components only.")
        GPU_AVAILABLE = False
# ADD GPU COMPONENTS HERE - END

# Dictionary mapping from string component name to actual class
COMPONENT_MAP = {
    # Imputers
    "SimpleImputer(strategy='mean')": cuSimpleImputer(strategy='mean') if GPU_AVAILABLE else SimpleImputer(strategy='mean'),
    "SimpleImputer(strategy='median')": cuSimpleImputer(strategy='median') if GPU_AVAILABLE else SimpleImputer(strategy='median'),
    "SimpleImputer(strategy='most_frequent')": cuSimpleImputer(strategy='most_frequent') if GPU_AVAILABLE else SimpleImputer(strategy='most_frequent'),
    "SimpleImputer(strategy='constant', fill_value=0)": cuSimpleImputer(strategy='constant', fill_value=0) if GPU_AVAILABLE else SimpleImputer(strategy='constant', fill_value=0),
    
    # Scalers
    "StandardScaler()": cuStandardScaler() if GPU_AVAILABLE else StandardScaler(),
    "MinMaxScaler()": cuMinMaxScaler() if GPU_AVAILABLE else MinMaxScaler(),
    "RobustScaler()": RobustScaler(), # No cuML equivalent
    "MaxAbsScaler()": MaxAbsScaler(), # No cuML equivalent
    "Normalizer()": Normalizer(), # No cuML equivalent
    
    # Encoders
    "OneHotEncoder(drop='first')": OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
    "OrdinalEncoder()": OrdinalEncoder(),
    
    # Feature selection
    "SelectKBest(k=5)": SelectKBest(k=5),
    "SelectKBest(k=10)": SelectKBest(k=10),
    "SelectPercentile(percentile=50)": SelectPercentile(percentile=50),
    "VarianceThreshold(threshold=0.1)": VarianceThreshold(threshold=0.1),
    
    # Feature transformation
    "PCA(n_components=2)": cuPCA(n_components=2) if GPU_AVAILABLE else PCA(n_components=2),
    "PCA(n_components=5)": cuPCA(n_components=5) if GPU_AVAILABLE else PCA(n_components=5),
    "PCA(n_components=10)": cuPCA(n_components=10) if GPU_AVAILABLE else PCA(n_components=10),
    "TruncatedSVD(n_components=5)": cuTruncatedSVD(n_components=5) if GPU_AVAILABLE else TruncatedSVD(n_components=5),
    "PolynomialFeatures(degree=2)": PolynomialFeatures(degree=2),
    
    # Classifiers
    "LogisticRegression(max_iter=1000)": cuLogisticRegression(max_iter=1000) if GPU_AVAILABLE else LogisticRegression(max_iter=1000),
    "LogisticRegression(max_iter=1000, C=0.1)": cuLogisticRegression(max_iter=1000, C=0.1) if GPU_AVAILABLE else LogisticRegression(max_iter=1000, C=0.1),
    "LogisticRegression(max_iter=1000, C=10)": cuLogisticRegression(max_iter=1000, C=10) if GPU_AVAILABLE else LogisticRegression(max_iter=1000, C=10),
    "DecisionTreeClassifier(max_depth=5)": DecisionTreeClassifier(max_depth=5),
    "DecisionTreeClassifier(max_depth=10)": DecisionTreeClassifier(max_depth=10),
    "DecisionTreeClassifier(max_depth=None)": DecisionTreeClassifier(max_depth=None),
    "RandomForestClassifier(n_estimators=50)": cuRandomForestClassifier(n_estimators=50) if GPU_AVAILABLE else RandomForestClassifier(n_estimators=50),
    "RandomForestClassifier(n_estimators=100)": cuRandomForestClassifier(n_estimators=100) if GPU_AVAILABLE else RandomForestClassifier(n_estimators=100),
    "RandomForestClassifier(n_estimators=200)": cuRandomForestClassifier(n_estimators=200) if GPU_AVAILABLE else RandomForestClassifier(n_estimators=200),
    "GradientBoostingClassifier(n_estimators=100)": GradientBoostingClassifier(n_estimators=100),
    "KNeighborsClassifier(n_neighbors=5)": cuKNeighborsClassifier(n_neighbors=5) if GPU_AVAILABLE else KNeighborsClassifier(n_neighbors=5),
    
    # Special token
    "END_PIPELINE": "END_PIPELINE"
}