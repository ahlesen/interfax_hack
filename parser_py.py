import requests
import numpy as np
import datetime
import re
from bs4 import BeautifulSoup

from time import sleep
import random
import datetime
import pandas as pd
import pickle
import os

from tqdm import tqdm_notebook, tqdm


def get_soup(name):
    """
    считаем 1 веб-страницу
    """
    response = requests.get(name)
    html = response.content
    soup = BeautifulSoup(html,'html.parser')
    return soup

def get_stories():
    """
    считаем все сюжеты с сайта интерфакс в таблицу (на момент написания кода было только 27 страниц)
    """
    df_all = pd.DataFrame()

    for i in tqdm(range(1,27)):
        mainpage = 'https://www.interfax.ru/story/page_'+str(i)
        soup = get_soup(mainpage)
        sleep(random.randint(5,10))
        list_stories = soup.find_all("div",{"class":"allStory"})
        
        titles = []
        links = []
        texts = []
        count = []
        
        for stories in list_stories: 
            for elem in stories.find_all("a",{"class":"img"}):
                if 'https' not in elem['href']:
                    links.append('https://www.interfax.ru'+elem['href'])
                else:
                    links.append(elem['href'])
                titles.append(elem['title'])
            for elem in stories.find_all("span",{"class":"text"}):
                texts.append(elem.text)
            for elem in stories.find_all("div",{"class":"info"}):
                count.append(elem.text)
        print(len(titles),len(links),len(texts),len(count))
        df_story = pd.DataFrame({'title':titles,
                            'link':links,
                            'text':texts[:len(links)],
                            'count':list(map(lambda x: int(x.split(' материалов')[0]),count))[:len(links)],
                            'lastUpd':list(map(lambda x: x.split('Обновлено ')[1],count))[:len(links)],
                            })
        df_all = df_all.append(df_story, ignore_index=True)
    
    df_all.to_excel('data/stories.xlsx', index=False)
    return df_all

def sport_check(link):
    """
    Спортивные страницы имеют другую структуру, парсятся иначе, поэтому ставим маркер
  
    """
    splitted_link = [i.split('-') for i in link.split('.')]
    splitted_link = ' '.join(list(np.concatenate(np.array(splitted_link))))

    splitted_link = splitted_link.split(' ')
    if 'sport' in splitted_link:
        marker = False
        
    else:

        marker = True

    return marker

def dataset_news_prep(stories:pd.DataFrame) -> pd.DataFrame:
    """
    считаем все новости с неспортивными сюжетами 
    """
    MAINPAGES = []
    MEGA_TITLES = []
    MEGA_TEXTS = []
    COUNTS = []
    LASTUpd = []
    LINKS = []
    TITLES = []
    TEXTS = []

    mainpages = len(stories)

    for page in tqdm_notebook(range(mainpages)):

        mainpage = stories.iloc[page]['link']
        mega_title = stories.iloc[page]['title']
        mega_text = stories.iloc[page]['text']
        count = stories.iloc[page]['count']
        lastUpd = stories.iloc[page]['lastUpd']
        
        response = requests.get(mainpage)
        html = response.content
        soup = BeautifulSoup(html,'html.parser')
        
        list_stories = soup.find_all("section",{"class":"chronicles__item"})
        

        for story in list_stories:
        
            try:
            
                link = story.find('a')['href']

            except:
                print('no link')
                continue

            else:    
                
                try :
                    title = story.find('a')['title']
                except:

                    print('no title')
                    continue

                else:

                    full_link = 'https://www.interfax.ru'+link

                    marker = sport_check(full_link)

                    if marker == False:
                        print('sport article')
                        continue

                    page_soap = get_soup(full_link)
                    text = page_soap.find_all("article",{"itemprop":"articleBody"})[0].text


                    MAINPAGES.append(mainpage)
                    MEGA_TITLES.append(mega_title)
                    MEGA_TEXTS.append(mega_text)
                    COUNTS.append(count)
                    LASTUpd.append(lastUpd)
                    LINKS.append(link)
                    TITLES.append(title)
                    TEXTS.append(text)
        sleep(2)

    
    full_stories_frame = pd.DataFrame({'mainpage':MAINPAGES,
                            'MEGA_TITLES':MEGA_TITLES,
                            'MEGA_TEXTS':MEGA_TEXTS,
                            'COUNTS':COUNTS,
                            'LASTUpd':LASTUpd,
                            'LINKS':LINKS,
                            'TITLES':TITLES,
                            'TEXTS':TEXTS
                        
                        })

    return full_stories_frame

def get_article_text(link:str) -> str:
    if 'https://www.sport' not in link:
        link =  'https://www.sport-interfax.ru/'+link
    page_soap = get_soup(link)
    text = page_soap.find_all("article",{"itemprop":"articleBody"})[0].text
    return text

def get_links_w_titles_texts(list_stories_:list, verbose=False, with_sleep=True):
    links = []
    titles = []
    texts = []
    for elem in tqdm_notebook(list_stories_):
        
        try:
            check2 = re.findall('(olymp2020/\d+|IceHockey2021/\d+|euro2020/\d+|\d+)',elem['href'])
            if (len(check2) >=1) :
                check_link = check2[0] 
                if len(check_link) >4:
                    if verbose:
                        print(1,check_link,elem)
                        print(elem['href'])
                    if len(elem.text) > 1:
                        if verbose:
                            print(3,elem.text)
                            print(20*'---')
                        link = elem['href']
                        try:
                            text = get_article_text(link)
                            texts.append(text)
                            if with_sleep:
                                sleep(random.randint(1,3))
                        except:
                            print('ERROR TEXT!',link)
                            continue
                        links.append(link)
                        titles.append(elem.text)
#                         return links, titles, texts
        except:
            print('ERROR:',elem)
            continue
    return links, titles, texts

def dataset_sport_news_prep(df_stories:pd.DataFrame) -> pd.DataFrame:
    """
    считаем все новости со спортивными сюжетами 
    """
    df_all = pd.DataFrame()

    for page in range(df_stories.shape[0]):
        soup = get_soup(df_stories['link'].iloc[page])
        list_stories_ = soup.find_all("a")
        
        links, titles, texts = get_links_w_titles_texts(list_stories_, with_sleep=True)
        
        df1 = pd.DataFrame({'link':links,'title':titles, 'text':texts})
        
        if df1.shape[0] <1:
            print('ERROR in parsing:',df_stories['title'].iloc[page])
        
        df1['mainpage'] = df_stories.iloc[page]['link']
        df1['mega_title'] = df_stories.iloc[page]['title']
        df1['mega_text'] = df_stories.iloc[page]['text']
        df1['count'] = df_stories.iloc[page]['count']
        df1['lastUpd'] = df_stories.iloc[page]['lastUpd']
        
        df1 = df1[['mega_title', 'mainpage', 'mega_text', 'count', 'lastUpd', 
            'link', 'title',  'text']]
        
        df_all = df_all.append(df1, ignore_index=True)
    
    return df_all 



def main():
    # проверка наличия таблицы со всеми сюжетами, если её нет, то запускаем парсинг
    flag_story = False
    for elem in os.listdir('data/'):
        if 'stories' in elem:
            flag_story = True
    if flag_story:
        df_stories = pd.read_excel('data/stories.xlsx')
    else:
        df_stories = get_stories()
    
    # в текущей версии мы для каждого сюжета парсим его новости только на первой таблице
    # в следующей итерации сделаем парсинг всех страниц

    # отдельно считываем новости только с сайта  interfax.ru
    df_interfax = dataset_news_prep(stories=df_stories) 
    # отдельно считываем новости только с сайта  sport-interfax.ru
    df_sport_interfax = dataset_sport_news_prep(stories=df_stories) 

    # сохранение получившихся таблиц
    df_interfax.to_excel('data/f_st.xlsx', index=False)
    df_sport_interfax.to_excel('data/all_sports.xlsx', index=False)