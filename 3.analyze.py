import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Импорт текста и меток к тексту в формате txt.
with open('data/text.txt', encoding='utf-8') as input:
    text = input.readlines()

with open('data/label.txt', encoding='utf-8') as input:
    labels = input.readlines()

X = text
y = labels

# Чистовая обработка текста, удаление числительных, единичных букв, пробелов и тд.
documents = []

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Удаление оставшихся спецсимволов.
    document = re.sub(r'\W', ' ', str(X[sen]))

    # Удаление одиночных букв.
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Замена нескольких пробелов подряд на один.
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Перевод в нижний регистр.
    document = document.lower()

    # Перевод в единственное число им. падежа.
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# Перевод в вектора.
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('russian'))
X = vectorizer.fit_transform(documents).toarray()

tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Разделение на обучающий и проверочный массивы.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Классификатор.
classifier = RandomForestClassifier(n_estimators=1500, random_state=0)

# Другие классификаторы, подставляются в classifier.
# make_pipeline(StandardScaler(), SVC(gamma='auto'))
# KNeighborsClassifier(n_jobs=-1, n_neighbors=3)
# DecisionTreeClassifier(criterion='gini', splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None)
# RandomForestClassifier(n_estimators=1000, random_state=0)

# Обучение.
classifier.fit(X_train, y_train)

# Предсказание.
y_pred = classifier.predict(X_test)

# Вывод метрик.
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# Pickle дамп модели.
with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

# Выгрузка модели обратно.
with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)

# Проверка выгруженной модели, имеет другой тег model.
y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2))

# Матрица зависимостей.
titles_options = [("Матрица зависимости для классификатора Decision Tree.", None)]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()