import requests
from bs4 import BeautifulSoup
import datetime

# file paths
horse_form_path = 'scraping_outputs/data_v2/horse_form/'

def get_month_num(month_str):
    if month_str == 'Jan':
        return '01'
    elif month_str == 'Feb':
        return '02'
    elif month_str == 'Mar':
        return '03'
    elif month_str == 'Apr':
        return '04'
    elif month_str == 'May':
        return '05'
    elif month_str == 'Jun':
        return '06'
    elif month_str == 'Jul':
        return '07'
    elif month_str == 'Aug':
        return '08'
    elif month_str == 'Sep':
        return '09'
    elif month_str == 'Oct':
        return '10'
    elif month_str == 'Nov':
        return '11'
    elif month_str == 'Dec':
        return '12'

def scrape_horse_form(page_url):
    page_request = requests.get(page_url)
    soup = BeautifulSoup(page_request.text, 'lxml')

    horse_form_dic = {}
    

    entire_form = soup.find_all('div', {'class' : "runner-histories"})
    results_list = entire_form[0].find_all('ul', {'class' : 'timeline'})

    for result in results_list:
        race_result_dic = {}
        
        
        res_divs = result.find_all('li', {'class' : 'timeline-cont'})

        placing = int(res_divs[0].find('span', {'class' : 'round-text formSummaryPosition'}).text)
        n_starters = int(res_divs[0].find('span', {'class' : 'starters'}).text)

        
        
        race_divs = res_divs[1].find_all('li', {'class' : 'timeline-disc'})
        race_date = race_divs[0].find('span', {'class' : 'date'}).text
        day, month, year = race_date.split('-')
        race_date_dt = datetime.date(int('20' + year), int(get_month_num(month)), int(day))

    
        print()
        print(race_date_dt)
        print()
    

def main():
    print('running')
    scrape_horse_form('https://www.punters.com.au/horses/sebrakate_657774/')

if __name__ == '__main__':
    main()