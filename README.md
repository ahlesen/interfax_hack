# парсинг
 - прежде всего мы осуществили парсинг сюжетов с новостями сайта интерфакс - parser_py.py

# обучение финальной модели
 - train_mT5.ipynb

# перед запуском модели нужно скачать веса и положить в папку mT5_checkpoint
 - ссылка: https://drive.google.com/file/d/16dNM9-YjA1o7enW14h2uXuOlyAxJ2D0o/view?usp=sharing

# пример запуска моделeй
 - baseline модель tfidf: python main_tf_idf.py --i 'data_interfax/dataset_public.json' --string_limit 30 --o preds_baseline.txt
 - mT5 модель (финальная на ней запускаемся): python main_mT5.py --i 'data_interfax/dataset_public.json'  --o preds_mT5.txt

 
