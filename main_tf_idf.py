import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import string
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

parser = argparse.ArgumentParser(description='argparse')
parser.add_argument("--i",  type=str, help="path to yout json file_name which we should read")
parser.add_argument("--o",  type=str, help="path to yout json file_name in which we should write")
parser.add_argument("--string_limit", default=30, type=int, help="limit len of predicted string")

# список наиболее популярных стоп слов
stop_words = pd.read_csv('stopwords.txt')
stop_words = list(stop_words['c'])
stop_words = stop_words + ['URL','USER','HASHTAG','ap','ар', '2ар', 'interfax', 'ru','ak']

# установка диапазона n-gram и колонки, которая используется для обучения модели
best_params = {
    'gram_range':(1,5),
    'text2model':'input_text'
}

# функция убирает стоп слова
def removeStopwords(wordlist, stopwords):
    return [w for w in (wordlist) if w not in stopwords]

translator = str.maketrans('', '', string.punctuation)


def preprocess_text(Text,punct=False, figures=False, stopwords = False):
    """
    На вход подается колонка с текстом, текст идет в нижний регистр убираются знаки препинания и сторонние символы,
    смайлы, хештеги, по команде убираются стоп слова и цифры
    """
    result=[]
    for sentense in tqdm(Text):
        string=(sentense.lower())
        string = [word for word in string.split() if word.startswith('@') == False]
        string = ' '.join([word for word in string if word.startswith('http') == False])
        
        string = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', string)
        string = re.sub('([>:;=-]+[\(cсCС/\\[L{<@]+)|([\(]{2,})', ' ', string)
        # # заменяем веселые/радостные смайлики на общий токен
        string = re.sub('([>:;=-]+[\)dD*pPрРbB]+)|([\)]{2,})', ' ', string)

    
        string = re.sub('[^a-zA-Zа-яА-Я0-9]+', ' ', string)
        
        
        
        if punct==True:
            string=string.translate(translator)
        if figures==True:
            string=re.sub(r'\d+', '', string)
            
        clear_sentence = string.split()
        if stopwords == True:
            clear_sentence = removeStopwords(clear_sentence,stop_words)
        
        
        
        
        clear_sentence = ' '.join(clear_sentence)
        

        
        result.append(clear_sentence)
    return result

def dataset_creation(data:list):
    """
    Происходит сборка датафрейма из json файла, т.к. обучение идет на заголовках, новости : молния автоматически становятся заголовками
    """
    df_val = pd.DataFrame()

    test_count = []

    for num_story in tqdm(range(len(data))):

        story  = data[num_story]
        test_count.append(len(story['news']))
        for num_news, elem in enumerate(story['news']):
            try:
                del elem['codes']
            except:
                  pass

            try:
                df1 = pd.DataFrame(elem, index=[0])
                df1['mega_title'] = story['title']
                df1['num_story'] = num_story
                df1['count'] = len(story['news'])
                df_val = df_val.append(df1,ignore_index=True)
            except:
                print(f'ERROR: num_story = {num_story} | num_news = {num_news} ')

    df_val.columns = ['id', 'lastUpd', 'type_id', 'language_id', 'slugline', 'title',
       'subheadline', 'trashline', 'dateline', 'text', 'background',
       'mega_title', 'num_story', 'count']

    return df_val

def count_tokens(text):
    """
    Подсчет токенов
    """
    text = text.split()
    cnt = 0
    for word in text:
        if word.isdigit():
            continue
        cnt+=1
    return cnt


def check_max_tfidf(mega_title, df_val, target_title,col_print, gram_range = (1,3), limit_string=30):
    """
    Расчет матрицы tf-idf

    mega_title: сюжет
    df_val: входной датафрейм
    target_title: название поля с сюжетами
    gram_range: range n-gram
    limit_string: трешхолд на размер предсказания

    Функция делает суммаризацию для 1 сюжета
    """
    df_check = df_val[df_val[target_title] == mega_title]

    corpus = df_check[col_print].values
    
    vectorizer = TfidfVectorizer(ngram_range=gram_range)
    X_check = vectorizer.fit_transform(corpus)
    df_tf_idf = pd.DataFrame(X_check.toarray(),columns=vectorizer.get_feature_names())

    max_values = df_tf_idf.max(axis=0)

    flag_2_next_max_value = True
    while flag_2_next_max_value:
        
        cols_w_max_values = max_values[max_values == np.max(max_values)]

        df_best = cols_w_max_values.to_frame().reset_index()
        df_best.columns = ['best_name','tfidf val']
        df_best['count_name'] = df_best['best_name'].apply(lambda x: len(x))
        df_best['count_tokens'] = df_best['best_name'].apply(lambda x: count_tokens(x))
        
        df_best.sort_values(['count_tokens'], ascending=False, inplace=True)

        best_col = df_best['best_name'].iloc[0]

        if (not best_col.isdigit()) | (df_best['count_tokens'].iloc[0] != 0 ) | (df_best['count_tokens'].iloc[0] > limit_string ):
            flag_2_next_max_value = False
        else:
            max_values.drop(best_col,inplace = True)
            if len(max_values) == 0:
                break
    return best_col, df_best

def make_predictions_tfidf(dataset, target_title, limit_string):
    """
    Предсказание для списка сюжетов
    """
    preds_baseline = []
    mega_titles = dataset[target_title].unique()
    for megatitle_num in tqdm(range(len(mega_titles))):
        mega_title = mega_titles[megatitle_num]

        best_col, _ = check_max_tfidf(mega_title,
                                                      dataset, 
                                                      target_title=target_title,
                                                      col_print = best_params['text2model'], 
                                                      gram_range = best_params['gram_range'],
                                                      limit_string=limit_string)
        preds_baseline.append(best_col)
    return preds_baseline

def main(file_name_in:str, string_limit:int, file_name_out:str,):
    """
    На вход подается название json файла, на выходе получаем список предсказанных сюжетов,
    который записывается в текстовый файл.

    Функция: - считывает файл
             - приводит его в датафрейм
             - производит препроцессинг
             - делает предсказание на основе tf-idf
             - производит запись в файл
    """
    with open(file_name_in,'r') as file:
        data = json.load(file)

    assert type(data) == list ,'Должны быть списки сюжетов'

    dataset = dataset_creation(data)
    dataset.loc[dataset[dataset['type_id'] == 'FLASH'].index,'title'] = dataset[dataset['type_id'] == 'FLASH']['text']
    dataset = dataset[['title','mega_title','text', 'num_story']]
    dataset = dataset.rename(columns={'title':'input_text','mega_title':'target_text','text':'text_all'})
    dataset['input_text'] = preprocess_text(dataset['input_text'], True, stopwords=True)

    dataset['target_text_prep'] = preprocess_text(dataset['target_text'], True, stopwords=False)

    dataset['text_all'] = preprocess_text(dataset['text_all'], True, stopwords=True)
    # убираем инфу до интерфакс (заглушка)
    dataset['text_all'] = dataset['text_all'].apply(lambda x: x.split('интерфакс',1)[1] if 'интерфакс' in x else x)

    # делаем предсказание для tfidf
    preds_baseline = make_predictions_tfidf(dataset,target_title='num_story',limit_string=string_limit)
    print(len(preds_baseline))
    with open(file_name_out, 'w+') as fout:
        for preds in preds_baseline: 
            fout.write(preds+'\n')
    return preds_baseline 

args = parser.parse_args()
main(args.i, args.string_limit, args.o)
# Идет моделирование!!!!!!!!
    