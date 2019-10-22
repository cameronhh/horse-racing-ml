import os
import copy
import sys
import pandas as pd
import numpy as np
import random

from sklearn import metrics
from sklearn import preprocessing

class BetManager:
    def __init(self):
        print()

    def implied_probability(self, betting_odds):
        return 1 / betting_odds

    def calculate_betting_outcomes(self, original_test_data, predictions):
        pred_matrix = pd.concat([original_test_data, predictions], axis=1)
        n_rows, n_cols = pred_matrix.shape

        pred_matrix.sort_values(['race_id', 'predicted_margin', 'Best Fixed Odds'], ascending=(True, True, False), inplace=True)
        pred_matrix.reset_index(drop=True, inplace=True)
        cur_rID = ''
        prev_rID = ''
        prediction_list = []
        for i in range(n_rows):
            prev_rID = cur_rID
            cur_rID = pred_matrix.loc[i, 'race_id']
            if cur_rID != prev_rID:
                prediction_list.append(1)
            else:
                prediction_list.append(0)
        pred_matrix = pd.concat([pred_matrix, pd.Series(prediction_list, name='prediction_label')], axis=1)                

        n_bets = 0
        n_bets_won = 0
        units_staked = 0.0
        total_returned = 0.0
        for i in range(n_rows):
            if pred_matrix.loc[i, 'prediction_label'] == 1:
                fixed_odds_paying = pred_matrix.loc[i, 'Best Fixed Odds']
                odds_paying = fixed_odds_paying
                stake = 1
                units_staked += stake
                n_bets += 1
                if pred_matrix.loc[i, 'win_label'] == 1:
                    n_bets_won += 1
                    total_returned += 1 * odds_paying

        roi = 0.0
        if units_staked > 0:
            roi = ((total_returned - units_staked)/units_staked)

        print("Single Win Betting outcomes:")
        print()
        print("  Horses bet on: " + str(n_bets))
        print("       Bets won: " + str(n_bets_won))
        print("   Units staked: " + str(units_staked))
        print("   Gross return: " + str(total_returned))
        print("     Net profit: " + str(total_returned - units_staked))
        print("            ROI: " + str(roi))
        print()