import requests
from bs4 import BeautifulSoup
import re

import pandas as pd
import os
import sys
import time
import datetime
from io import StringIO


class Scraper:
    def __init__(self, storage_path):
        self.site_domain = "https://www.punters.com.au"
        self.csv_url = "https://www.punters.com.au/form-guide/spreadsheet-"
        self.output_file_path = storage_path

    def _get_races(self, soup):
        return soup.find_all('table', attrs={'class':'results-table'})

    def _total_seconds(self, time_str):
        values = re.split('\:', time_str)
        if len(values) == 1:
            return float(values[0])
        if len(values) == 2:
            return (float(values[0]) * 60) + float(values[1])

    def scrape_single_event(self, page_url):
        print("Starting Scrape of Single Event from: " + page_url)
        print()
        page_request = requests.get(page_url)
        soup = BeautifulSoup(page_request.text, 'lxml')

        all_race_results_tables = self._get_races(soup)
        all_race_times = [] # list of 1st place times for races on this page
        race_numbers = [] # list of unique ids for races on this page

        for i in range(len(all_race_results_tables)):
            table = all_race_results_tables[i]
            table_rows = table.find_all('tr')
            race_times = []
            winner_finish_time = 0.0

            try:
                # For each row (horse)
                for j in range(len(table_rows)):
                    row = table_rows[j]
                    if j == 0: # if it is the header row, get the race id
                        race_numbers.append(row.find_all('th')[0].get('data-eventid'))

                    cells_in_row = row.find_all('td')
                    if len(cells_in_row) != 0 and j != 0:
                        if j == 1: # if it the winning horse, get the winning time
                            winner_finish_time = self._total_seconds(cells_in_row[4].text.strip())
                            race_times.append(winner_finish_time)
                        else:
                            race_times.append(winner_finish_time + (float(cells_in_row[4].text.strip()[:-1]) / 6))
                all_race_times.append(race_times)

                print("    Downloading CSV for race " + race_numbers[i])
                race_form_url = self.csv_url + race_numbers[i]
                csv = requests.get(race_form_url, stream=True)

                race_data = pd.read_csv(StringIO(csv.text), index_col=False)
                n_rows, n_cols = race_data.shape

                if (len(table_rows) - 1) == n_rows: # if no horses were scratced
                    race_data.sort_values('Finish Result (Updates after race)', ascending=True, inplace=True)
                    race_data.reset_index(drop=True, inplace=True)
                    race_data["Finish Result (Updates after race)"] = pd.Series(all_race_times[i])
                    output_file_name = race_numbers[i] + ".csv"
                    race_data.to_csv(path_or_buf=self.output_file_path+output_file_name, index=None)
                else:
                    print("      Race skipped as some scratched horses haven't been removed.")
            except:
                e = sys.exc_info()[0]
                print("      Encountered exception: " + str(e))

    def scrape_events_from_track(self, page_url):
        print("Starting Scrape of Track Events from: " + page_url)
        print()
        page_request = requests.get(page_url)
        soup = BeautifulSoup(page_request.text, 'lxml')

        past_events_list = soup.find_all('li', "latest-result")
        for li in past_events_list:
            race_day_link = li.find('a')['href']
            event_page_url = self.site_domain + race_day_link
            self.scrape_single_event(page_url=event_page_url)

    def scrape_events_from_date(self, date_string): # format "yyyy-mm-dd"
        page_url = "https://www.punters.com.au/racing-results/" + date_string + "/"
        print("Starting Scrape of Events on " + date_string + " from " + page_url)
        print()
        page_request = requests.get(page_url)
        soup = BeautifulSoup(page_request.text, 'lxml')

        race_labels_list = soup.find_all('ul', attrs={'class':'jump-to__results-list'})[0].find_all('li')
        for li in race_labels_list:
            if '(' in li.get_text(): # if it is an international race
                continue
            race_day_link = li.find('a')['href']
            event_page_url = self.site_domain + race_day_link
            print(event_page_url)
            self.scrape_single_event(event_page_url)

    def scrape_event_between_dates(self, date1, date2):
        ''' ensure date 1 < date2
        '''
        year, month, day = date1.split("-")
        d1 = datetime.date(int(year), int(month), int(day))
        year, month, day = date2.split("-")
        d2 = datetime.date(int(year), int(month), int(day))
        timedelta = datetime.timedelta(hours=24)

        while d1 > d2:
            self.scrape_events_from_date(d1.isoformat())
            d1 = d1 - timedelta
