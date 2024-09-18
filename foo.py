import pandas as pd
from typing import List, Dict, Any

class PipelineSensor:
    def __init__(self, model_info_csv: str):
        self.model_info = pd.read_csv(model_info_csv)

    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        analysis = {
            'num_samples': len(data),
            'num_features': len(data.columns) - 1,
            'data_type': self._infer_data_type(data),
            'linearity': self._check_linearity(data)
        }
        return analysis
    # def suggest_models(self, data_analysis: Dict[str, Any], task: str) -> List[str]:
    #     compatible_models = []
    #     for _, model in self.model_info.iterrows():
    #         if (str(model['Task Performed']).lower() == (task).lower() and
    #         (model['Type of Data']).lower() == (data_analysis['data_type']).lower()):
    #             if data_analysis['linearity'] == 'linear':
    #                 if model['Performance with Linear data'] in ['Excellent', 'Good']:
    #                     compatible_models.append(model['Model'])
    #             else:
    #                 if model['Performance with Non linear data'] in ['Excellent', 'Good']:
    #                     compatible_models.append(model['Model'])
        
    #     if not compatible_models:
    #         print(f"No compatible models found for task: {task} and data type: {data_analysis['data_type']}")
        
        return compatible_models


    def validate_pipeline(self, pipeline: List[str], task: str) -> bool:
        for model_name in pipeline:
            model_info = self.model_info[self.model_info['Model'] == model_name]
            if model_info.empty:
                print(f"Model not found in CSV: {model_name}")
                return False
            
            if str(model_info.iloc[0]['Task Performed']).lower() != str(task).lower():
                print(f"Incompatible task: {model_name} performs {model_info.iloc[0]['Task Performed']}, but the required task is {task}")
                return False
        
        return True

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        model_info = self.model_info[self.model_info['Model'] == model_name]
        if model_info.empty:
            print(f"Model not found in CSV: {model_name}")
            return {}
        return model_info.iloc[0].to_dict()



    def _infer_data_type(self, data: pd.DataFrame) -> str:
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > len(data.columns) / 2:
            return 'Numeric'
        else:
            return 'Categorical'

    def _check_linearity(self, data: pd.DataFrame) -> str:
        return 'linear' if data.shape[1] < 5 else 'non-linear'

if __name__ == "__main__":
    sensor = PipelineSensor('model_info.csv')
    
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [3, 6, 9, 12, 15]
    })
    
    analysis = sensor.analyze_data(data)
    print("Data Analysis:", analysis)
    
    # suggested_models = sensor.suggest_models(analysis, task='Regression')
    # print("Suggested Models:", suggested_models)
    
    pipeline = ['Linear Regression', 'Polynomial Regression']
    is_valid = sensor.validate_pipeline(pipeline, task='Regression')
    print("Is Pipeline Valid:", is_valid)
    for i in pipeline:
        model_details = sensor.get_model_details(i)
        print("Model Details:", model_details)