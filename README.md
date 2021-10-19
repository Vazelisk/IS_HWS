### Поисковик на движках `Bert`, `FastText`, `BM25`, `TF IDF`, `Countvectorizer`.<br>
### В качестве данных используются вопросы/ответы Mail.ru про любовь.<br>
### Веб-версия сделана с помощью `Streamlit`.<p>

Для работы нужны следующие файлы:<p>
Корпус - https://www.kaggle.com/bobazooba/thousands-of-questions-about-love<p>
FastText - https://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz<p>

Пример работы:<p>
![image](https://github.com/Vazelisk/IS_HWS/blob/master/example.gif)

### Инструкция
`streamlit_app.py` запускает веб-приложение. До этого необходимо единожды запустить search_engine, чтобы создать все необходимые файлы, покажет скоринг и запустит поиск в терминале.

Пример выдачи скрипта:<p>
![image](https://user-images.githubusercontent.com/42929213/137906245-d96a1542-4d2c-4fc5-a6eb-9ccbdc06b2ac.png)
