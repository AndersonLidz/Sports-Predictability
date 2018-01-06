# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:17:03 2018

@author: Acer
"""

from sklearn.linear_model import LogisticRegression
from random import choices, gauss, seed
from itertools import compress
import pandas as pd
import numpy as np
import time
data = pd.read_csv(r"C:\Users\Acer\Documents\Python Scripts\sports\nfl2018.csv")
print(data.head())

data.columns = list(map(lambda x: x.strip(), data.columns))

#need dictionary of key:teams, value:game index's

team_ind_dict = {team: list(compress([i for i in range(data.shape[0])],list(map(lambda i: data['team_home'].iloc[i]==team or 
                                      data['team_away'].iloc[i]==team, range(data.shape[0]))))) for team in data['team_home'].unique()}
print(team_ind_dict)

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
#maybe add weighting to this
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

#method is sum or mean
def run_scorer(the_dataset, order, method = 'sum'):
    if method=='sum':
        return([np.sum(list(map(lambda x: scorer(x, the_dataset), mutual_indexs(order,the_dataset, i)))) for i in range(the_dataset.shape[0])])

    if method=='mean':
        return([np.mean(list(map(lambda x: scorer(x, the_dataset), mutual_indexs(order,the_dataset, i)))) for i in range(the_dataset.shape[0])])


seed(321)
potentials = ['Titans', 'Chiefs','Falcons', 'Rams','Bills', 'Jaguars','Panthers', 'Saints','Eagles', 'Patriots', 'Steelers', 'Vikings']
seeds = [5, 4, 6, 3, 6, 3, 5, 4, 1, 1, 2, 2]
potentials_sb_wins= [0]*len(potentials)
placement = {str(potentials[i]):i for i in range(len(potentials))}
sb_attempts= [0]*len(potentials)
data_restart = data
num_simulations = 3
for sim in range(num_simulations):
    init_time = time.time()
    data = data_restart
    reg = LogisticRegression()
    X_full = np.array(run_scorer(data, 4, method = 'sum'))
    bools = (X_full!=0)
    X = X_full[bools]
    Y_full = np.array(list(map(int,map(lambda i: data['score_home'].iloc[i]>data['score_away'].iloc[i], range(data.shape[0])))))
    Y = Y_full[bools]
    X.shape = (-1,1)
    reg.fit(X, Y)

    Wildcards = [('Titans', 'Chiefs'),('Bills', 'Jaguars'),('Falcons', 'Rams'),('Panthers', 'Saints')]
    Fixed = ['Patriots', 'Steelers', 'Eagles','Vikings']
    
    R1 = {'a':Wildcards[0], 'b':Wildcards[1], 'c':Wildcards[2], 'd':Wildcards[3]}
    R1_winners = {'a':None, 'b':None, 'c':None, 'd':None}
    R2_winners = {'a':None, 'b':None, 'c':None, 'd':None}
    R3_winners = {'a':None, 'b':None}
    R4_winners = {'a':None}

    R2 = {'a':(Fixed[0],R1_winners['a']), 'b':(Fixed[1],R1_winners['b']), 'c':(Fixed[2],R1_winners['c']), 'd':(Fixed[3],R1_winners['d'])}

    R3 = {'a':(R2_winners['b'],R2_winners['a']), 'b':(R2_winners['d'],R2_winners['c'])}

    R4 = {'a':(R3_winners['b'],R3_winners['a'])}
    for pair in [[R1,R1_winners], [R2,R2_winners], [R3,R3_winners], [R4,R4_winners]]:

        data_to_add = pd.DataFrame(columns = data.columns)
        current_round = pair[0]
        if current_round==R2:

            afc = [R1_winners['a'], R1_winners['b']]
            highest_afc = afc[np.argmin([seeds[placement[R1_winners['a']]],seeds[placement[R1_winners['b']]]])]
            lowest_afc = afc[np.argmax([seeds[placement[R1_winners['a']]],seeds[placement[R1_winners['b']]]])]
            nfc = [R1_winners['c'],R1_winners['d']]
            highest_nfc = nfc[np.argmin([seeds[placement[R1_winners['c']]],seeds[placement[R1_winners['d']]]])]
            lowest_nfc = nfc[np.argmax([seeds[placement[R1_winners['c']]],seeds[placement[R1_winners['d']]]])]
            current_round = {'a':(Fixed[0],lowest_afc), 'b':(Fixed[1],highest_afc), 'c':(Fixed[2],lowest_nfc), 'd':(Fixed[3],highest_nfc)}

        elif current_round==R3:

            current_round = {'a':(R2_winners['b'],R2_winners['a']), 'b':(R2_winners['d'],R2_winners['c'])}
        elif current_round==R4:

            current_round = {'a':(R3_winners['b'],R3_winners['a'])}
            sb_attempts[placement[current_round['a'][0]]]+=1
            sb_attempts[placement[current_round['a'][1]]]+=1
            
        round_winners = pair[1]

        for j in current_round.keys():
    
            temp_data_accum = data

            temp_data_accum = temp_data_accum.append(pd.DataFrame([[current_round[j][0], current_round[j][1], 0, 0, 0]], columns=data.columns), ignore_index=True)

            obs = temp_data_accum.shape[0] - 1

            games_indexs = mutual_indexs(4,temp_data_accum, game_index = obs)

            game_score = np.sum(list(map(lambda x: scorer(x, temp_data_accum), games_indexs)))

            winner = choices([1,0], weights = reg.predict_proba(game_score)[0])[0]
            round_winners.update({str(j):current_round[j][winner]})
            
            score_home = 0
            score_away = 0
            if winner==1:
                while score_home>=score_away:
                    score_home = round(gauss(np.mean(data['score_home']), np.std(data['score_home'])))
                    score_away = round(gauss(np.mean(data['score_away']), np.std(data['score_away'])))
                data_to_add = data_to_add.append(pd.DataFrame([[current_round[j][0],current_round[j][1], score_home,score_away, score_home-score_away]], columns = data.columns),ignore_index=True)
            else:
                while score_home<=score_away:
                    score_home = round(gauss(np.mean(data['score_home']), np.std(data['score_home'])))
                    score_away = round(gauss(np.mean(data['score_away']), np.std(data['score_away'])))
                data_to_add = data_to_add.append(pd.DataFrame([[current_round[j][0],current_round[j][1],score_home,score_away, score_home - score_away]], columns = data.columns), ignore_index=True)
        pair[1] = round_winners
        data = data.append(data_to_add, ignore_index=True)
        team_ind_dict = {team: list(compress([i for i in range(data.shape[0])],list(map(lambda i: data['team_home'].iloc[i]==team or 
                                          data['team_away'].iloc[i]==team, range(data.shape[0]))))) for team in data['team_home'].unique()}
        reg = LogisticRegression()
        X_full = np.array(run_scorer(data, 4, method = 'sum'))
        bools = (X_full!=0)
        X = X_full[bools]
        Y_full = np.array(list(map(int,map(lambda i: data['score_home'].iloc[i]>data['score_away'].iloc[i], range(data.shape[0])))))
        Y = Y_full[bools]
        X.shape = (-1,1)
        Y.shape = (-1,1)
        reg.fit(X, Y)

    potentials_sb_wins[placement[R4_winners['a']]]+=1
    print('current iteration',sim)
    print('iteration time', time.time()-init_time)


print(potentials)
print(potentials_sb_wins)
print(sb_attempts)