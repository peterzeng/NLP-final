import pandas as pd
from sklearn.model_selection import train_test_split

processed = pd.read_csv("full data.csv")
# sample = processed.sample(frac=0.1)
# print(type(sample))
# sample.to_csv("test.csv", header=True, index=False)
train, test = train_test_split(processed, test_size=0.3)

# print(train['Author'])
train_authors = list(set(train['Author']))
test_authors = list(set(test['Author']))

# train = train.drop(train[train['Author'] not in test_authors])
# test = test.drop(test[test['Author'] not in train_authors])
# # training_indices_not_in_test = train[ (train["Author"] not in test_authors) ].index
# # testing_indices_not_in_train = test[(test["Author"] not in train_authors)].index

# # train = train.drop(training_indices_not_in_test, inplace = True)
# # test = test.drop(testing_indices_not_in_train, inplace = True)

# train.to_csv('train.csv', header=True, index=False)
# test.to_csv('test.csv', header=True, index=False)
training_indices_not_in_test = []
testing_indices_not_in_train = []

for index, row in train.iterrows():
    # print(row['Author'])
    if row.loc['Author'] not in test_authors:
        # print(row.loc['Author'])
        # print(row['Author'])
        # print(index)
        training_indices_not_in_test.append(index)

train.drop(training_indices_not_in_test, inplace = True)

for index, row in test.iterrows():
    # print(row['Author'])
    if row.loc['Author'] not in train_authors:
        testing_indices_not_in_train.append(index)

test.drop(testing_indices_not_in_train, inplace = True)

print(len(list(set(train['Author']))))
print(len(list(set(test['Author']))))
train.to_csv('train.csv', header=True, index=False)
test.to_csv('test.csv', header=True, index=False)