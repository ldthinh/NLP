# Using decision tree to predict disease
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# load data
data = pd.read_csv("Manual-Data/Training.csv")
df = pd.DataFrame(data)

cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
# split dataset into data_train and data_test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
# start training
load_model = DecisionTreeClassifier()
train_model = load_model.fit(x_train, y_train)

# acurracy of model
arcc = train_model.score(x_test, y_test)
print("Acurracy of model", arcc)
# Test
# Load data test
data_test = pd.read_csv("Manual-Data/Testing.csv")
datatest_df = pd.DataFrame(data_test)

symptoms_cols = datatest_df.columns
cols_data_test = symptoms_cols[:-1]
data_test_x = datatest_df[cols_data_test]
data_test_y = datatest_df['prognosis']
print(type(data_test_x))
# results = train_model.predict([data_test_x.iloc[0]])
# probal = train_model.predict_proba([data_test_x.iloc[0]])
results = train_model.predict(data_test_x)
# check result
for i in range(len(results)):
    if data_test_y[i] != results[i]:
        print('Pred: {0} Actual:{1}'.format(results[i], data_test_y[i]))
        print(i)








