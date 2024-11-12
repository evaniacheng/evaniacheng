import numpy as np
import pandas as pd
import sys

class NaiveBayes:
    def fit(self, X, y):
        n, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #mean, var (for numerical), and priors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)

        #for categorical features
        self._cat_probs = {}

        #determine if a feature is numerical or categorical
        self._categorical_features = []
        self._numerical_features = []
        for i, col in enumerate(X.columns):
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                self._categorical_features.append(i)
            else:
                self._numerical_features.append(i)

        #calculate mean, var, and categorical probablities
        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._priors[idx] = X_c.shape[0] / float(n)

            #handle numerical features: compute mean and variance
            for i in self._numerical_features:
                self._mean[idx,i] = X_c.iloc[:,i].mean()
                self._var[idx,i] = X_c.iloc[:,i].var() + 1e-6

            #compute categorical probabilities
            self._cat_probs[c] = {}
            for i in self._categorical_features:
                categories = X.iloc[:,i].unique()
                category_counts = X_c.iloc[:,i].value_counts()
                total = X_c.shape[0]
                self._cat_probs[c][i] = {
                    cat: (category_counts.get(cat,0)+1) / (total+len(categories)) for cat in categories
                }

    def predict(self, X):
        y_pred = [self._predict(x) for _, x in X.iterrows()]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = 0

            #gaussian likelihood for numerical features
            for i in self._numerical_features:
                mean = self._mean[idx,i]
                var = self._var[idx,i]
                class_conditional += np.log(self._pdf(mean, var, x.iloc[i]))
            
            #likelihood for categorical features
            for i in self._categorical_features:
                category_prob = self._cat_probs[c][i].get(x.iloc[i], 1e-6) #using smalll value for unseen categories
                class_conditional += np.log(category_prob)

            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, mean, var, x):
        numerator = np.exp(- (x-mean)**2 /  (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
def load_data(train_data_path, validate_data_path):
    #load training and test data
    train_data = pd.read_csv(train_data_path)
    validate_data = pd.read_csv(validate_data_path)

    #seperate features and labels
    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]
 
    X_test = validate_data

    return X_train, y_train, X_test

if __name__ == "__main__":
    train_data_path = sys.argv[1]
    validate_data_path = sys.argv[2]

    X_train, y_train, X_test = load_data(train_data_path, validate_data_path)

    model = NaiveBayes()
    model.fit(X_train, y_train)

    #predict on test data
    y_pred = model.predict(X_test)

    #print predictions
    for prediction in y_pred:
        print(1 if prediction == 1 else 0)