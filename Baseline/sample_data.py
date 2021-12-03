import pandas as pd
from sklearn.model_selection import train_test_split

processed = pd.read_csv("data.csv")

train, test = train_test_split(processed, test_size=0.3)

train_authors = list(set(train['Author']))
test_authors = list(set(test['Author']))

training_indices_not_in_test = []
testing_indices_not_in_train = []

for index, row in train.iterrows():
    if row.loc['Author'] not in test_authors:
        training_indices_not_in_test.append(index)

train.drop(training_indices_not_in_test, inplace = True)

for index, row in test.iterrows():
    if row.loc['Author'] not in train_authors:
        testing_indices_not_in_train.append(index)

test.drop(testing_indices_not_in_train, inplace = True)

# print(len(list(set(train['Author']))))
# print(len(list(set(test['Author']))))
# train.to_csv('train.csv', header=True, index=False)
# test.to_csv('test.csv', header=True, index=False)

# train_dataframe = pd.read_csv("train.csv")
# test_dataframe = pd.read_csv("test.csv")

train_dataframe = pd.DataFrame(train)
test_dataframe = pd.DataFrame(test)

auth_sort = sorted(train_dataframe['Author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
train_dataframe['Author_num'] = train_dataframe['Author'].map(swap_dict)

auth_sort = sorted(test_dataframe['Author'].unique())
dictOfAuthors = { i : auth_sort[i] for i in range(0, len(auth_sort) ) }
swap_dict = {value:key for key, value in dictOfAuthors.items()}
test_dataframe['Author_num'] = test_dataframe['Author'].map(swap_dict)

train_dataframe = train_dataframe.drop(columns="Author")
test_dataframe = test_dataframe.drop(columns="Author")

list_to_choose_train = train_dataframe.text.apply(lambda x : len(x)) > 0 
train_df_article = train_dataframe[list_to_choose_train]
list_to_choose_test = test_dataframe.text.apply(lambda x : len(x)) > 0 
test_df_article = test_dataframe[list_to_choose_test]

train_df_article.to_csv('train.csv', index = False)
test_df_article.to_csv('test.csv', index = False)
