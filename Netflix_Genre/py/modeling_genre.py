from netflix_genre import df_model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 

# logistic regression

# x = all the columns except main_genre
# y = main_genre (the label we want to predict)

x = df_model.drop('main_genre', axis = 1)
y = df_model['main_genre']

# 80% training set, 20% test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

# Initialize the model
logreg = LogisticRegression(max_iter=1000)

# train (fit) the model
logreg.fit(x_train, y_train)

# predict on test data
y_pred = logreg.predict(x_test)

# evaluate
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))