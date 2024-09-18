class MakePipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        X_transformed = X
        for step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed,y)
        X_transformed = step[-1].fit(X_transformed,y)

    def predict(self, X):
        X_transformed = X
        for step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)
        return self[-1].predict(X_transformed)