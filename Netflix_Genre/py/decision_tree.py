from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report 
from modeling_genre import x_train, y_train, x_test, y_test

# initialize
tree = DecisionTreeClassifier(max_depth=10, random_state = 42)

# Train
tree.fit(x_train, y_train)

#predict
y_pred_tree = tree.predict(x_test)

print("Decision Tree Accuracy Score:", accuracy_score(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))