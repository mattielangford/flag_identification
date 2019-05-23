import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv('flags.csv', header = 0)
# Data from http://archive.ics.uci.edu/ml/datasets/Flags
#print(flags.columns)
#print(flags.head(10))

labels = flags[["Landmass"]]
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

scores = []
#Tuning the model
for i in range(1, 21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  tree.fit(train_data, train_labels)
  score = tree.score(test_data, test_labels)
  scores.append(score)

plt.plot(range(1, 21), scores)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title('A Measure of Tree Depth on Performance')
plt.show()
