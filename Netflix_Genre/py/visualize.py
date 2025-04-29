import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from netflix_genre import df
from modeling_genre import x
from decision_tree import x_train, x_test, y_train, y_test, y_pred_tree, tree


# Plot genre distribution
plt.figure(figsize=(12,6))
sns.countplot(y='main_genre', data=df, order=df['main_genre'].value_counts().index)
plt.title('Genre Distribution')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_tree, labels=tree.classes_)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
plt.figure(figsize=(12,12))
disp.plot(xticks_rotation=90, cmap='Blues')
plt.title('Confusion Matrix for Decision Tree')
plt.show()


# Feature importances
feature_importances = pd.DataFrame({
    'feature': x.columns,
    'importance': tree.feature_importances_
}).sort_values(by='importance', ascending=False)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Also print
print(feature_importances)


