import numpy
import pandas
import joblib
import sklearn.metrics
import sklearn.ensemble
import sklearn.inspection
import matplotlib.pyplot


dataset = pandas.read_excel("Traumatismo craneoencefálico severo PEDIA.xlsx")

dataset = dataset.replace("NO", 0)
dataset = dataset.replace("SÍ", 1)
dataset = dataset.replace("ANORMAL", 0)
dataset = dataset.replace("NORMAL", 1)
dataset = dataset.replace("FEM", 0)
dataset = dataset.replace("MASC", 1)
dataset = dataset.replace("FALLA", 0)
dataset = dataset.replace("ÉXITO", 1)
dataset = dataset.replace("NR", numpy.nan)

dataset = dataset.dropna()

x_true = dataset.iloc[:, :-1]
y_true = dataset.iloc[:, -1:]

features = x_true.columns
target = y_true.columns


classifier = sklearn.ensemble.RandomForestClassifier()
classifier.fit(x_true, y_true)

y_pred = classifier.predict(x_true)

joblib.dump(classifier, "classifier.joblib")


accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

permutation = sklearn.inspection.permutation_importance(classifier, x_true, y_true)
importances = permutation.importances_mean / sum(permutation.importances_mean)

for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.2f}")


figure, axes = matplotlib.pyplot.subplots(figsize=(8.5, 8.5))

axes.bar(features, importances)
axes.set_xticklabels(features, rotation=90)
axes.set_xlabel("Característica")
axes.set_ylabel("Importancia")
axes.set_title(f"Traumatismo Craneoencefálico Severo (Pediátrico)\nInteligencia Artificial con Precisión de {accuracy*100:.2f}%")
axes.grid(True)

figure.tight_layout()
figure.savefig("importances.png")
