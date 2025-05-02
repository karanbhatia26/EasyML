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
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.decomposition import KernelPCA

COMPONENT_MAP = {
    # Imputers
    "SimpleImputer(strategy='mean')": SimpleImputer(strategy='mean'),
    "SimpleImputer(strategy='median')": SimpleImputer(strategy='median'),
    "SimpleImputer(strategy='most_frequent')": SimpleImputer(strategy='most_frequent'),
    "SimpleImputer(strategy='constant', fill_value=0)": SimpleImputer(strategy='constant', fill_value=0),
    
    # Scalers
    "StandardScaler()": StandardScaler(),
    "MinMaxScaler()": MinMaxScaler(),
    "RobustScaler()": RobustScaler(),
    "MaxAbsScaler()": MaxAbsScaler(),
    "Normalizer()": Normalizer(),
    
    # Encoders
    "OneHotEncoder(drop='first')": OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
    "OrdinalEncoder()": OrdinalEncoder(),
    
    # Feature selection
    "SelectKBest(k=5)": SelectKBest(k=5),
    "SelectKBest(k=10)": SelectKBest(k=10),
    "SelectPercentile(percentile=50)": SelectPercentile(percentile=50),
    "VarianceThreshold(threshold=0.1)": VarianceThreshold(threshold=0.1),
    
    # Feature transformation
    "PCA(n_components=2)": PCA(n_components=2),
    "PCA(n_components=5)": PCA(n_components=5),
    "PCA(n_components=10)": PCA(n_components=10),
    "TruncatedSVD(n_components=5)": TruncatedSVD(n_components=5),
    "PolynomialFeatures(degree=2)": PolynomialFeatures(degree=2),
    
    # Additional preprocessors
    "QuantileTransformer(output_distribution='normal')": QuantileTransformer(output_distribution='normal'),
    "PowerTransformer()": PowerTransformer(),
    
    # Additional feature transformers
    "Nystroem(kernel='rbf', n_components=100)": Nystroem(kernel='rbf', n_components=100),
    "RBFSampler(gamma=0.1, n_components=100)": RBFSampler(gamma=0.1, n_components=100),
    "KernelPCA(n_components=50, kernel='rbf')": KernelPCA(n_components=50, kernel='rbf'),
    
    # Classifiers
    "LogisticRegression(max_iter=1000)": LogisticRegression(max_iter=1000),
    "LogisticRegression(max_iter=1000, C=0.1)": LogisticRegression(max_iter=1000, C=0.1),
    "LogisticRegression(max_iter=1000, C=10)": LogisticRegression(max_iter=1000, C=10),
    "DecisionTreeClassifier(max_depth=5)": DecisionTreeClassifier(max_depth=5),
    "DecisionTreeClassifier(max_depth=10)": DecisionTreeClassifier(max_depth=10),
    "DecisionTreeClassifier(max_depth=None)": DecisionTreeClassifier(max_depth=None),
    "RandomForestClassifier(n_estimators=50)": RandomForestClassifier(n_estimators=50),
    "RandomForestClassifier(n_estimators=100)": RandomForestClassifier(n_estimators=100),
    "RandomForestClassifier(n_estimators=200)": RandomForestClassifier(n_estimators=200),
    "GradientBoostingClassifier(n_estimators=100)": GradientBoostingClassifier(n_estimators=100),
    "KNeighborsClassifier(n_neighbors=5)": KNeighborsClassifier(n_neighbors=5),
    
    # Additional classifiers
    "SVC(kernel='rbf', C=10)": SVC(kernel='rbf', C=10, probability=True),
    "SVC(kernel='poly', degree=3)": SVC(kernel='poly', degree=3, probability=True),
    "LinearSVC(max_iter=2000)": LinearSVC(max_iter=2000, dual=False),
    "MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    "AdaBoostClassifier()": AdaBoostClassifier(),
    "HistGradientBoostingClassifier()": HistGradientBoostingClassifier(),
    
    # Special token
    "END_PIPELINE": "END_PIPELINE"
}