import os
import copy
import sys
import pandas as pd
import numpy as np
import random

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RandomUnderSampler, NearMiss

class DataManager:
    def __init__(self, path, win_type):
        self.training_file_path = path
        self.win_type = win_type

    def _encode_gender(self, dataset):
        ''' Must encode gender data before splitting up the dataset '''
        gender_dummies = pd.get_dummies(dataset["Gender"], prefix="gender", drop_first=True)
        leading_cols = dataset.iloc[:, 0:6]
        future_scale_cols = dataset.iloc[:, 7:10]
        trailing_cols = dataset.iloc[:, 10:]
        return pd.concat([leading_cols, future_scale_cols, gender_dummies, trailing_cols], axis=1)

    def _load_dataset(self, file_path, win_type, split):
        dataset = pd.read_csv(file_path)
        dataset = self._encode_gender(dataset)
        dataset.sample(frac=1).reset_index(drop=True, inplace=True) # this is a shuffled data set

        unique_raceID_list = dataset["race_id"].unique()
        n_races = len(unique_raceID_list)
        n_test_races = int(n_races * split)
        random.shuffle(unique_raceID_list)
        test_raceID_list = unique_raceID_list[0:n_test_races]
        test_data = dataset.loc[dataset["race_id"].isin(test_raceID_list)]
        training_data = dataset.loc[dataset["race_id"].isin(test_raceID_list) == 0]
        #training_data = dataset.loc[dataset['Finish Result (Updates after race)'] < 25]

        print("  number of races: " + str(n_races))
        print("number of results: " + str(len(dataset)))
        print()
        print("  training races: " + str(n_races - n_test_races))
        print("training results: " + str(len(training_data)))
        print()
        print("      test races: " + str(abs(n_test_races)))
        print("    test results: " + str(len(test_data)))
        print()

        leading_column_offset = 2
        X_train = training_data.iloc[:, leading_column_offset:-3]
        X_test = test_data.iloc[:, :-3] # no leading offset because we want to save the horse names and race ID

        regr_train = training_data.iloc[:, -3:-2]
        regr_test = test_data.iloc[:, -3:-2]
        win_train = training_data.iloc[:, -2:-1]
        win_test = test_data.iloc[:, -2:-1]
        place_train = training_data.iloc[:, -1:]
        place_test = test_data.iloc[:, -1:]
        
        shuffle_seed = random.randint(0, 100)
        X_train = X_train.sample(frac=1, random_state=shuffle_seed)
        regr_train = regr_train.sample(frac=1, random_state=shuffle_seed)
        win_train = win_train.sample(frac=1, random_state=shuffle_seed)
        place_train = place_train.sample(frac=1, random_state=shuffle_seed)
        shuffle_seed = random.randint(0, 100)
        X_test = X_test.sample(frac=1, random_state=shuffle_seed)
        regr_test = regr_test.sample(frac=1, random_state=shuffle_seed)
        win_test = win_test.sample(frac=1, random_state=shuffle_seed)
        place_test = place_test.sample(frac=1, random_state=shuffle_seed)

        # save the testing data with horse name and race_ids after it has been shuffled
        X_test.reset_index(drop=True, inplace=True)
        regr_test.reset_index(drop=True, inplace=True)
        win_test.reset_index(drop=True, inplace=True)
        place_test.reset_index(drop=True, inplace=True)
        og_test_data = pd.concat([X_test, regr_test, win_test, place_test], axis=1)
        self.original_test_data = copy.deepcopy(og_test_data)

        X_test = X_test.iloc[:, 2:]
        if win_type == 0:
                y_train = regr_train
                y_test = regr_test
        elif win_type == 1:
            y_train = win_train
            y_test = win_test
        elif win_type == 2:
            y_train = place_train
            y_test = place_test

        return X_train, X_test, y_train, y_test

    def get_original_test_data(self):
        return self.original_test_data

    def load(self, train_test_split):
        X_train, X_test, y_train, y_test = self._load_dataset(self.training_file_path, self.win_type,
        train_test_split)
        # do scaling here if wanted

        return X_train, X_test, y_train, y_test

    def balance(self, X, y, method):
        if method == 'smote':
            sampler = SMOTE(random_state=1, ratio=0.4)
            X, y = sampler.fit_sample(X, y)
        elif method == 'enn':
            sampler = EditedNearestNeighbours(sampling_strategy='not minority', n_neighbors=1)
            X, y = sampler.fit_sample(X, y)
        elif method == 'tomeks':
            sampler = TomekLinks(sampling_strategy='not minority', n_jobs=-1)
            X, y = sampler.fit_sample(X, y)
        elif method == 'uniform-undersampling':
            sampler = RandomUnderSampler(sampling_strategy=0.35)
            X, y = sampler.fit_sample(X, y)
        elif method == 'uniform-oversampling':
            sampler = RandomOverSampler(sampling_strategy=0.3)
            X, y = sampler.fit_sample(X, y)
        elif method == 'near-miss':
            sampler = NearMiss(sampling_strategy='not minority', n_neighbors=5, version=1)
            X, y = sampler.fit_sample(X, y)
        else:
            print("DataManager balance(): did not recognise method: " + method)
            sys.exit()
        return X, y

    def unscale(self, X):
        ''' performs an inverse transform on parameter X
            probably throws an error if the original data given to this model
            was not scaled as self.scaler will not be defined
        '''
        return self.scaler.inverse_transform(X)