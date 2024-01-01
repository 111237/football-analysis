import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import warnings
from mplsoccer import Pitch
from scipy.stats import binned_statistic_2d
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')

df = pd.DataFrame()
for i in range(13):
    file_name = 'events_England_' + str(i+1) + '.json'
    path = os.path.join(r'C:\Users\99662\Desktop\wyscout', file_name)
    with open(path) as f:
        data = json.load(f)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
next_event = df.shift(-1, fill_value=0)
df["nextEvent"] = next_event["subEventName"]
df['kickedOut'] = df.apply(lambda x:1 if x.nextEvent == "Ball out of the field" else 0, axis=1)
move_df = df.loc[df['subEventName'].isin(['Simple pass', 'High pass', 'Head pass', 'Smart pass', 'Cross'])]
delete_passes = move_df.loc[move_df['kickedOut'] == 1]
move_df = move_df.drop(delete_passes.index)
move_df['x'] = move_df.positions.apply(lambda cell: cell[0]['x']*105/100)
move_df['y'] = move_df.positions.apply(lambda cell: (100-cell[0]['y'])*68/100)
move_df['end_x'] = move_df.positions.apply(lambda cell: cell[1]['x']*105/100)
move_df['end_y'] = move_df.positions.apply(lambda cell: (100-cell[1]['y'])*68/100)
move_df = move_df.loc[(((move_df["end_x"] != 0) & (move_df["end_y"] != 68)) & ((move_df["end_x"] != 105) & (move_df["end_y"] != 0)))]
pitch = Pitch(line_color='black', pitch_type='custom', pitch_width=68, pitch_length=105, line_zorder=2)
move = pitch.bin_statistic(move_df.x, move_df.y, statistic='count', bins=(16, 12), normalize=False)
fig, ax =pitch.grid(grid_height=0.9, title_height=0.06, axis=False, endnote_height=0.04, title_space=0, endnote_space=0)
pcm = pitch.heatmap(move, cmap='Blues', edgecolor='grey', ax=ax['pitch'])
ax_cbar = fig.add_axes((0.93, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Moving actions 2D histogram', fontsize=30)
plt.show()
move_count = move["statistic"]


shot_df = df.loc[df['subEventName'] == 'Shot']
shot_df['x'] = shot_df.positions.apply(lambda cell: cell[0]['x']*105/100)
shot_df['y'] = shot_df.positions.apply(lambda cell: (100-cell[0]['y'])*68/100)
pitch = Pitch(line_color='Black', pitch_type='custom', pitch_width=68, pitch_length=105, line_zorder=2)
shot = pitch.bin_statistic(shot_df.x, shot_df.y, statistic='count', bins=(16,12), normalize=False)
fig, ax =pitch.grid(grid_height=0.715,grid_width=0.95, endnote_height=0.065, endnote_space=0.01, title_height=0.15, title_space=0.01,axis=False)
pcm = pitch.heatmap(shot, cmap='Greens', edgecolor='grey', ax=ax['pitch'])
ax_cbar=fig.add_axes((0.93, 0.093, 0.03, 0.786))
cbar=plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Shots 2D histogram', fontsize = 30)
plt.show()
shot_count = shot['statistic']

goal_df = shot_df.loc[shot_df.apply(lambda x: {'id': 101} in x.tags, axis=1)]
goal = pitch.bin_statistic(goal_df.x, goal_df.y, statistic='count', bins=(16, 12), normalize=False)
fig, ax =pitch.grid(grid_height=0.715,grid_width=0.95, endnote_height=0.065, endnote_space=0.01, title_height=0.15, title_space=0.01,axis=False)
pcm = pitch.heatmap(goal, cmap='Reds',edgecolor= 'grey', ax=ax['pitch'])
ax_cbar=fig.add_axes((0.94,0.093,0.03,0.786))
cbar=plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Goals 2D histogram', fontsize = 30)
plt.show()
goal_count = goal['statistic']
move_df["start_sector"] = move_df.apply(lambda row: tuple([i[0] for i in binned_statistic_2d(np.ravel(row.x), np.ravel(row.y),
                                                               values ="None", statistic="count",
                                                               bins=(16, 12), range=[[0, 105], [0, 68]],
                                                               expand_binnumbers=True)[3]]), axis = 1)
move_df['end_sector'] = move_df.apply(lambda row:tuple(i[0] for i in binned_statistic_2d(np.ravel(row.end_x), np.ravel(row.end_y),
                                                                values='None', statistic='count',
                                                                bins=(16, 12), range=[[0, 105], [0, 68]],
                                                                expand_binnumbers=True)[3]), axis=1)
df_count_starts = move_df.groupby(['start_sector'])['eventId'].count().reset_index()
df_count_starts.rename(columns={'eventId': 'count_starts'}, inplace=True)

transition_matrices = []
for i, row in df_count_starts.iterrows():
    start_sector = row['start_sector']
    count_starts = row['count_starts']
    #get all events that started in this sector
    this_sector = move_df.loc[move_df["start_sector"] == start_sector]
    df_cound_ends = this_sector.groupby(["end_sector"])["eventId"].count().reset_index()
    df_cound_ends.rename(columns = {'eventId':'count_ends'}, inplace=True)
    T_matrix = np.zeros((12, 16))
    for j, row2 in df_cound_ends.iterrows():
        end_sector = row2["end_sector"]
        value = row2["count_ends"]
        T_matrix[end_sector[1] - 1][end_sector[0] - 1] = value
    T_matrix = T_matrix / count_starts
    transition_matrices.append(T_matrix)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

#Change the index here to change the zone.
goal["statistic"] = transition_matrices[80]
pcm  = pitch.heatmap(goal, cmap='Reds', edgecolor='grey', ax=ax['pitch'])
#legend to our plot
ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
cbar = plt.colorbar(pcm, cax=ax_cbar)
fig.suptitle('Transition probability for one of the middle zones', fontsize = 30)
plt.show()

shot_probability = shot_count/(shot_count+move_count)
goal_probability = goal_count/shot_count
goal_probability[np.isnan(goal_probability)] = 0
move_probability = move_count/(shot_count+move_count)
transition_matrices_array = np.array(transition_matrices)
xT = np.zeros((12, 16))
for i in range(5):
    shoot_expected_payoff = goal_probability*shot_probability
    move_expected_payoff = move_probability*(np.sum(np.sum(transition_matrices_array*xT, axis = 2), axis = 1).reshape(16,12).T)
    xT = shoot_expected_payoff + move_expected_payoff


    fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.01, title_space=0, endnote_space=0)
    goal["statistic"] = xT
    pcm  = pitch.heatmap(goal, cmap='Oranges', edgecolor='grey', ax=ax['pitch'])
    labels = pitch.label_heatmap(goal, color='blue', fontsize=9,
                             ax=ax['pitch'], ha='center', va='center', str_format="{0:,.2f}", zorder = 3)

    ax_cbar = fig.add_axes((1, 0.093, 0.03, 0.786))
    cbar = plt.colorbar(pcm, cax=ax_cbar)
    txt = 'Expected Threat matrix after ' +  str(i+1) + ' moves'
    fig.suptitle(txt, fontsize = 30)
    plt.show()
successful_moves = move_df.loc[move_df.apply(lambda x: {'id': 1801} in x.tags, axis=1)]
successful_moves["xT_added"] =successful_moves.apply(lambda row: xT[row.end_sector[1]-1][row.end_sector[0]-1]-xT[row.start_sector[1]-1][row.start_sector[0]-1])
value_adding_actions  = successful_moves.loc[successful_moves['xT_added'] > 0]
xT_by_player = value_adding_actions.groupby(['playerId'])['xT_added'].sum().reset_index()
path = r'C:\Users\99662\Desktop\Wyscout\players.json'
player_df = pd.read_json(path,encoding='unicode-escape')
player_df.rename(columns = {'wyId':'playerId'}, inplace=True)
breakpoint()