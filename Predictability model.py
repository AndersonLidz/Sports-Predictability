# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:55:42 2018

@author: Acer
"""

import pandas as pd
import numpy as np
import pickle
from itertools import compress
from sklearn.linear_model import LogisticRegression
output_accuracy_list = []
nfl = pd.read_csv(r"C:\summer-sport-files\nfl2017.csv")
nba = pd.read_csv(r"C:\summer-sport-files\nba2017.csv")
mlb = pd.concat([pd.read_csv(r"C:\summer-sport-files\mlb2017.csv"),pd.read_csv(r"C:\summer-sport-files\mlb2017-1.csv")], ignore_index= True)
for data, pick_file in zip([nfl,nba, mlb],['nfl_accuracy.pickle','nba_accuracy.pickle','mlb_accuracy.pickle']):
    output_accuracy_list = []
    print(data.head())
    data = data.iloc[:nfl.shape[0]]

    data.columns = list(map(lambda x: x.strip(), data.columns))

    #creates dictionary of Teams (key) and List of indices where team played (value)
    team_ind_dict = {team: list(compress([i for i in range(data.shape[0])],list(map(lambda i: data['team_home'].iloc[i]==team or 
                                          data['team_away'].iloc[i]==team, range(data.shape[0]))))) for team in data['team_home'].unique()}
    print(team_ind_dict)
    #input: order integer, the_dataset as pandas dataframe, game_index integer
    #output: List of lists, first indice for each list is the current game index, other indices in list form sequence of mutual teams
            # connecting the home team and away team.
    def mutual_indexs(order, the_dataset, game_index=0, current_recursion=0, temp_list = [], temp_team = [], ret_list = []):
        if current_recursion==0:
            return(mutual_indexs(order, the_dataset, game_index, current_recursion+1, [game_index], temp_team = the_dataset['team_home'].iloc[game_index], ret_list = []))
        if current_recursion< order:
            next_indices = list(compress(team_ind_dict[temp_team], 
                                         (list(map(lambda x: x<game_index and x not in temp_list, team_ind_dict[temp_team])))))

            new_temp = [temp_list]*len(next_indices)

            tuples = list(zip(new_temp,next_indices))

            new_temp = list(map(lambda x: x[0]+[x[1]], tuples))
            for k in range(len(new_temp)):
                temp_team_new = next(filter(lambda x: x!=temp_team, 
                                       [the_dataset['team_home'].iloc[new_temp[k][current_recursion]], the_dataset['team_away'].iloc[new_temp[k][current_recursion]]]))

                mutual_indexs(order, the_dataset, game_index, current_recursion+1, new_temp[k], temp_team_new, ret_list)

        elif temp_team == the_dataset['team_away'].iloc[game_index]: 
            ret_list.append(temp_list)

        return(ret_list)
    
    score_difs = np.array(data['score_home']) - np.array(data['score_away'])
    data['score_difs'] = np.array(data['score_home']) - np.array(data['score_away'])
    #Input: list_index: a single list from the list of lists produced by mutual_indexs function.
    #       the_dataset: Pandas DataFrame
    #       Weighting: Boolean
    #Output: Integer, sum of score differences as specified in appendix:[1]
    def scorer(list_index, the_dataset, weighting = False):
        team_oriented = the_dataset['team_home'].iloc[list_index[0]]
        score = 0
        if weighting==False:
            for i in list_index[1:]:
                if team_oriented==the_dataset['team_home'].iloc[i]:
                    score+= the_dataset['score_difs'].iloc[i]
                    team_oriented = the_dataset['team_away'].iloc[i]
                else:
                    score-= the_dataset['score_difs'].iloc[i]
                    team_oriented = the_dataset['team_home'].iloc[i]
        else:
            for i in list_index[1:]:
                if team_oriented==the_dataset['team_home'].iloc[i]:
                    score+= (the_dataset['score_home'].iloc[i] - the_dataset['score_away'].iloc[i])*(i/list_index[0])
                    team_oriented = the_dataset['team_away'].iloc[i]
                else:
                    score+= (the_dataset['score_away'].iloc[i] - the_dataset['score_home'].iloc[i])*(i/list_index[0])
                    team_oriented = the_dataset['team_home'].iloc[i]
        return(score)

    #input: the_dataset: pandas dataframe, order:integer (number mutual teams + 2), method:'Sum' or 'Mean'
    #output: mutual_indexs scored by appendix [1] for every observation in dataset.
    def run_scorer(the_dataset, order, method = 'sum'):
        if method=='sum':
            return([np.sum(list(map(lambda x: scorer(x, the_dataset), mutual_indexs(order,the_dataset, i)))) for i in range(the_dataset.shape[0])])
        
        if method=='mean':
            return([np.mean(list(map(lambda x: scorer(x, the_dataset), mutual_indexs(order,the_dataset, i)))) for i in range(the_dataset.shape[0])])
    
    #evaluate predictive results for each dataset with differing numbers of mutual teams
    #outputs pickle file of list of output accuracy's for logistic regression model
    for k in range(3,7):

        X_full = np.array(run_scorer(data, k, method = 'sum'))
        #train model on data only satisfying non-zero criterion (ie predictor has found at least one sequence of mutual teams
        #that connects the home and away team)
        bools = (X_full!=0)

        X = X_full[bools]
        
        test_train_split = round(.7*len(X))
        print(test_train_split, len(X), 'train test split')
        
        X_train = X[0:test_train_split]
        X_test = X[test_train_split:nfl.shape[0]]
        
        Y_full = np.array(list(map(int,map(lambda i: data['score_home'].iloc[i]>data['score_away'].iloc[i], range(data.shape[0])))))
        Y = Y_full[bools]
        Y_train = Y[0:test_train_split]
        Y_test = Y[test_train_split:nfl.shape[0]]

        reg = LogisticRegression()

        X_train.shape = (-1,1)
        X_test.shape = (-1,1)
        reg.fit(X_train, Y_train)

        output_accuracy = np.sum(reg.predict(X_test)==Y_test)/len(Y_test)
        print(output_accuracy, 'percent correct')
        output_accuracy_list.append(output_accuracy)

    pickle.dump(output_accuracy_list, open(pick_file, 'wb'))