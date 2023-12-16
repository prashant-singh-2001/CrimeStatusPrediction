import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv("Cleaned_Data.csv")

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)

rf = RandomForestClassifier(
    n_estimators=10, criterion='entropy', max_features=10, class_weight='balanced')

kf = KFold(n_splits=10, random_state=7, shuffle=True)

result = cross_val_score(rf, X_resampled, y_resampled, cv=kf)

print(result.mean())

print('%0.8f' % result.var())

print(result.max())

print(result.min())

print('%0.8f' % result.std())
