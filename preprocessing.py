import pandas as pd
import sys
import os
import multiprocessing
import math
from sklearn import preprocessing

def split_list(alist, n_parts):
    length = len(alist)
    return [alist[i*length // n_parts: (i+1)*length // n_parts] for i in range(n_parts)]

def num_placers(n_horses):
    if n_horses < 5:
        return 1
    elif n_horses < 8:
        return 2
    else:
        return 3

def get_class(prize_money): # proprietary classes for races based on prize money
    if prize_money < 21000:
        return "class4"
    elif prize_money < 36000:
        return "class3"
    elif prize_money < 76000:
        return "class2"
    else:
        return "class1"

class DataProcessor:
    columns_to_remove = [4, 5, 6, 9, 11, 12, 14, 15, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51,
    52, 53, 54, 56, 57, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123,
    124, 125, 126, 127]

    def __init__(self, n_jobs=8):
        self.input_data_path = "scraping_outputs/new_races/"
        self.output_data_path = "data/"
        self.temp_data_path = "scraping_outputs/bin/"
        self.archive_data_path = "scraping_outputs/processed_races/"
        self.n_jobs = n_jobs

    def _add_to_file(self, data_frame, file_name, path_to_file):
        full_file_path = path_to_file + file_name
        with open(full_file_path, 'a') as f:
            data_frame.to_csv(f, header=(f.tell()==0), index=None)

    def _swap_columns(self, data_frame, name_1, name_2):
        cols = list(data_frame.columns)
        a, b = cols.index(name_1), cols.index(name_2)
        cols[b], cols[a] = cols[a], cols[b]
        return data_frame[cols]

    def _smoothing_function(self, r, N, r_avg):
        return ((1 - math.exp(-0.7 * N)) * r) + (math.exp(-0.7 * N) * r_avg)

    ### DO FIRST Remove cols not needed for processing SEPARATE FROM LIST
    def _processing_remove_cols(self, race_data):
        race_data.drop(race_data.columns[self.columns_to_remove], axis=1, inplace=True)
        return race_data

    ### Insert column for unique race id NEED TO BE SEPARATE FROM FUNCTION LIST
    def _processing_race_id(self, race_data, file_name):
        race_data.insert(0, "race_id", [file_name[:-4] for i in range(race_data.shape[0])])
        return race_data

    ### Insert column for the number of horses in the race
    def _processing_num_horses(self, race_data):
        race_data.insert(3, "num_horses", [race_data.shape[0] for i in range(race_data.shape[0])])
        return race_data

    def _processing_race_distance(self, race_data, race_distance):
        race_data.insert(8, "race_distance", [race_distance for i in range(race_data.shape[0])])
        return race_data

    ### Append columns for win and place labels
    def _processing_sample_labels(self, race_data):
        win_label_list = [1] + [0 for i in range(race_data.shape[0]-1)]
        n_placers = num_placers(race_data.shape[0])
        place_label_list = [1 for i in range(n_placers)] + [0 for i in range(race_data.shape[0]-n_placers)]
        race_data["win_label"] = pd.Series(win_label_list)
        race_data["place_label"] = pd.Series(place_label_list)
        return race_data

    def _processing_clean_sparse(self, race_data):
        ''' Addresses data sparsity/inconsistencies noticed in race form guides:
            Adds "first_start" feature {1, 0}
            'Interpolates' last start margin if horse is first starter
            Sets last start prize money == 0 for any first starter
            Sets last start prize money == 0 for any horse who did not win prize
                money in their previous race
            Sets last start margin == 0 for any horse that won their previous race
        '''
        n_horses, n_cols = race_data.shape
        first_start_label = []
        for i in range(n_horses):
            if race_data.loc[i, "Career Runs"] < 1:
                first_start_label.append(1)
                race_data.loc[i, "Last Start Margin"] = 4 # an 'interpolated' value to prevent sparsity
                race_data.loc[i, "Last Start Prize Money"] = 0
            else:
                first_start_label.append(0)
                if pd.isnull(race_data.loc[i, "Last Start Prize Money"]):
                    race_data.loc[i, "Last Start Prize Money"] = 0
            if race_data.loc[i, "Last Start Finish Position"] == 1:
                race_data.loc[i, "Last Start Margin"] = 0
        return race_data, first_start_label

    ### Remove 'm' from end of string in Last Start Distance column
    def _processing_last_start_distance(self, race_data):
        for i, trial in race_data.iterrows():
            race_data.loc[i, "Last Start Distance"] = race_data.loc[i, "Last Start Distance"][:-1]
        return race_data

    ### Label encoding of categorical feature: horse num
    def _processing_horse_num(self, race_data):
        le = preprocessing.LabelEncoder()
        le.fit(race_data["Num"])
        race_data["Num"] = le.transform(race_data["Num"])
        return race_data

    def _processing_remove_strike_rate_noise(self, race_data):
        ''' Smooths the noise created by varying numbers of horse, jockey
            and trainer runs.
        '''
        # CSR_mean = race_data["Career Strike Rate"].mean()
        # CPSR_mean = race_data["Career Place Strike Rate"].mean()
        # DTSR_mean = race_data["Dry Track Strike Rate"].mean()
        # WTSR_mean = race_data["Wet Track Strike Rate"].mean()
        # TCSR_mean = race_data["This Condition Strike Rate"].mean()
        # TCPSR_mean = race_data["This Condition Place Strike Rate"].mean()
        # APM_mean = race_data["Average Prize Money"].mean()
        # JAHE_mean = race_data["Jockey Last 100 Avg Horse Earnings"].mean()
        # JSR_mean = race_data["Jockey Last 100 Strike Rate"].mean()
        # JPSR_mean = race_data["Jockey Last 100 Place Strike Rate"].mean()
        # TAHE_mean = race_data["Trainer Last 100 Avg Horse Earnings"].mean()
        # TSR_mean = race_data["Trainer Last 100 Strike Rate"].mean()
        # TPSR_mean = race_data["Trainer Last 100 Place Strike Rate"].mean()

        CSR_mean = 0.2
        CPSR_mean = 0.4
        DTSR_mean = 0.2
        WTSR_mean = 0.2
        TCSR_mean = 0.2
        TCPSR_mean = 0.4
        APM_mean = race_data["Average Prize Money"].mean()
        JAHE_mean = race_data["Jockey Last 100 Avg Horse Earnings"].mean()
        JSR_mean = race_data["Jockey Last 100 Strike Rate"].mean()
        JPSR_mean = race_data["Jockey Last 100 Place Strike Rate"].mean()
        TAHE_mean = race_data["Trainer Last 100 Avg Horse Earnings"].mean()
        TSR_mean = race_data["Trainer Last 100 Strike Rate"].mean()
        TPSR_mean = race_data["Trainer Last 100 Place Strike Rate"].mean()

        for i in range(race_data.shape[0]):
            race_data.loc[i, "Career Strike Rate"] = self._smoothing_function(race_data.loc[i, "Career Strike Rate"], race_data.loc[i, "Career Runs"], CSR_mean)
            race_data.loc[i, "Career Place Strike Rate"] = self._smoothing_function(race_data.loc[i, "Career Place Strike Rate"], race_data.loc[i, "Career Runs"], CPSR_mean)
            race_data.loc[i, "Dry Track Strike Rate"] = self._smoothing_function(race_data.loc[i, "Dry Track Strike Rate"], race_data.loc[i, "Dry Track Runs"], DTSR_mean)
            race_data.loc[i, "Wet Track Strike Rate"] = self._smoothing_function(race_data.loc[i, "Wet Track Strike Rate"], race_data.loc[i, "Wet Track Runs"], WTSR_mean)
            race_data.loc[i, "This Condition Strike Rate"] = self._smoothing_function(race_data.loc[i, "This Condition Strike Rate"], race_data.loc[i, "This Condition Runs"], TCSR_mean)
            race_data.loc[i, "This Condition Place Strike Rate"] = self._smoothing_function(race_data.loc[i, "This Condition Place Strike Rate"], race_data.loc[i, "This Condition Runs"], TCPSR_mean)
            race_data.loc[i, "Average Prize Money"] = self._smoothing_function(race_data.loc[i, "Average Prize Money"], race_data.loc[i, "Career Runs"], APM_mean)
            race_data.loc[i, "Jockey Last 100 Avg Horse Earnings"] = self._smoothing_function(race_data.loc[i, "Jockey Last 100 Avg Horse Earnings"], race_data.loc[i, "Jockey Last 100 Starts"], JAHE_mean)
            race_data.loc[i, "Jockey Last 100 Strike Rate"] = self._smoothing_function(race_data.loc[i, "Jockey Last 100 Strike Rate"], race_data.loc[i, "Jockey Last 100 Starts"], JSR_mean)
            race_data.loc[i, "Jockey Last 100 Place Strike Rate"] = self._smoothing_function(race_data.loc[i, "Jockey Last 100 Place Strike Rate"], race_data.loc[i, "Jockey Last 100 Starts"], JPSR_mean)
            race_data.loc[i, "Trainer Last 100 Avg Horse Earnings"] = self._smoothing_function(race_data.loc[i, "Trainer Last 100 Avg Horse Earnings"], race_data.loc[i, "Trainer Last 100 Starts"], JAHE_mean)
            race_data.loc[i, "Trainer Last 100 Strike Rate"] = self._smoothing_function(race_data.loc[i, "Trainer Last 100 Strike Rate"], race_data.loc[i, "Trainer Last 100 Starts"], JSR_mean)
            race_data.loc[i, "Trainer Last 100 Place Strike Rate"] = self._smoothing_function(race_data.loc[i, "Trainer Last 100 Place Strike Rate"], race_data.loc[i, "Trainer Last 100 Starts"], JPSR_mean)
        return race_data

    def _processing_remove_wet(self, race_data):
        if race_data.loc[0, "This Condition Runs"] != race_data.loc[0, "Dry Track Runs"]:
            print("race aborted, wet conditions")
            return pd.DataFrame()
        else:
            return race_data

    ### returning finish result to lengths instead of time
    def _processing_finish_time(self, race_data, distance):
        winner_finish_time = race_data.loc[0, "Finish Result (Updates after race)"]
        race_data["Finish Result (Updates after race)"] = (race_data["Finish Result (Updates after race)"] - winner_finish_time) * 6
        return race_data

    def _process_race_file(self, input_file_path, file_name, p_id):
        race_data = pd.read_csv(input_file_path)
        race_data = race_data.sort_values(["Finish Result (Updates after race)"], ascending=True)
        class_str = get_class(int(race_data.loc[1, "Prize Money"])) # get now because prize money col is dropped

        try: # An exception here infers that the winning horse is a first starter,
             # we consider a race like this to be an anomaly in the data, unless
             # every runner in the race is a first starter, in which case we
             # have too much sparsity anyway, so we drop the race
            race_distance = int((race_data.loc[0, "Last Start Distance"][:-1])) - race_data.loc[1, "Last Start Distance Change"]
        except:
            print("Winner was a first starter, dropping race...")
            return 1

        #race_data = self._processing_remove_wet(race_data)
        race_data, first_start_label = self._processing_clean_sparse(race_data)
        race_data = self._processing_remove_strike_rate_noise(race_data)
        race_data = self._processing_remove_cols(race_data)
        race_data = self._processing_race_id(race_data, file_name)
        race_data = self._swap_columns(race_data, "Horse Name", "Num")
        race_data = self._processing_num_horses(race_data)
        race_data = self._processing_sample_labels(race_data)
        #race_data = self._processing_horse_num(race_data)
        race_data = self._processing_race_distance(race_data, race_distance)
        race_data = self._processing_finish_time(race_data, race_distance)
        race_data.insert(2, "first_start", first_start_label)
        race_data = self._swap_columns(race_data, "num_horses", "Num")
        race_data = self._swap_columns(race_data, "Barrier", "Career Strike Rate")
        race_data = self._swap_columns(race_data, "Career Place Strike Rate", "Best Fixed Odds")
        race_data = self._swap_columns(race_data, "race_distance", "Career ROI")

        if race_data.isnull().values.any():
            return 2

        # scale the data
        leading_cols = race_data.iloc[:, :11]
        trailing_cols = race_data.iloc[:, -3:]
        pre_scaling_cols = race_data.iloc[:, 11:-3].astype(float)
        col_headers = list(pre_scaling_cols)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        scaler.fit(pre_scaling_cols)
        scaled_cols = scaler.transform(pre_scaling_cols)
        scaled_cols_df = pd.DataFrame(data=scaled_cols, columns=col_headers)
        race_data = pd.concat([leading_cols, scaled_cols_df, trailing_cols], axis=1)

        class_file_name = str(p_id) + "_" + class_str + "_master.csv"
        master_file_name = str(p_id) + "_" + "master.csv"

        self._add_to_file(race_data, master_file_name, self.temp_data_path)
        self._add_to_file(race_data, class_file_name, self.temp_data_path)
        return 0

    def _process_list_of_files(self, files_list, process_id):
        print(str(process_id) + "_id process starting...")
        for file_name in files_list:
            if file_name == ".DS_Store":
                continue
            else:
                # Get the new file name
                input_file_path = self.input_data_path + file_name
                output_file_path = self.output_data_path + file_name
                #print(str(process_id) + "_id Processing file: " + input_file_path)
                #print()
                result = self._process_race_file(input_file_path, file_name, process_id)
                if result == 0:
                    # Move the new file
                    os.rename(input_file_path, self.archive_data_path+file_name)
                    #print(str(process_id) + "_id " + input_file_path + " moved to " + self.archive_data_path+file_name)
                    #print()
                elif result == 2:
                    print("still some sparsity after processing in race: " + str(file_name))
                    os.rename(input_file_path, self.archive_data_path+file_name)
                else:
                    # File was unusable, but it may not always be unusable, so we archive it
                    os.rename(input_file_path, self.archive_data_path+file_name)
                    #print(str(process_id) + "_id " + input_file_path + " archived, was unusable for these processing parameters")
                    #print()
        print(str(process_id) + "_id process completed...")

    def _join_master_files(self):
        temp_file_list = os.listdir(self.temp_data_path)
        for file_name in temp_file_list:
            if file_name == ".DS_Store":
                continue
            else:
                output_file_name = file_name[2:] # getting rid of p_id in name
                file_data = pd.read_csv(self.temp_data_path + file_name)
                self._add_to_file(file_data, output_file_name, self.output_data_path)
                print("Deleting temporary file: " + file_name)
                os.remove(self.temp_data_path + file_name)


    def process_new_training_data(self):
        new_files = os.listdir(self.input_data_path)
        print("Processing Training Data with " + str(self.n_jobs) + " processes:")
        new_files_split = split_list(new_files, self.n_jobs)
        processes = []

        for i in range(self.n_jobs):
            _process = multiprocessing.Process(target=self._process_list_of_files,
            args=(new_files_split[i], i))
            _process.start()
            processes.append(_process)

        for _process in processes:
            _process.join()

        print("All processes completed, joining individual process master files...")
        self._join_master_files()
        print("Done.")



if __name__ == '__main__':
    data_processor = DataProcessor(n_jobs=16)
    data_processor.process_new_training_data()