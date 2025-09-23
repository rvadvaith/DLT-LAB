from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


data = load_wine()
X, y = data.data, data.target

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
dtree = DecisionTreeClassifier()


scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}


for clf, name in [(logreg, "Logistic Regression"), (dtree, "Decision Tree")]:
    scores = cross_validate(clf, X, y, cv=5, scoring=scoring)
    print(f"\n{name} average performance (5-fold CV):")
    print(f"Accuracy:  {scores['test_accuracy'].mean():.3f}")
    print(f"Precision: {scores['test_precision'].mean():.3f}")
    print(f"Recall:    {scores['test_recall'].mean():.3f}")
    print(f"F1 Score:  {scores['test_f1'].mean():.3f}")
