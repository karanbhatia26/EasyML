def get_dataset_name_from_id(data_id):
    """Fetch dataset name from OpenML API using data_id"""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=data_id, as_frame=True)
    return data.details['name']

# Test function to print all dataset names
def print_dataset_names():
    dataset_ids = {
        'credit-g': 31,
        'travel': 45065,
        'banknote': 1462,
        'click-prediction': 4134
    }
    
    print("Dataset Names from OpenML IDs:")
    print("-" * 40)
    
    for key, data_id in dataset_ids.items():
        try:
            name = get_dataset_name_from_id(data_id)
            print(f"ID {data_id} ({key}): {name}")
        except Exception as e:
            print(f"ID {data_id} ({key}): Error - {str(e)}")
    
    print("-" * 40)
if __name__ == "__main__":
    print_dataset_names()