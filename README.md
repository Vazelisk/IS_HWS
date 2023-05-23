### Search Engine using `Bert`, `FastText`, `BM25`, `TF IDF`, `Countvectorizer`.<br>
### As answer data I use questions/answers from Service "Crowd answer users questions" by Mail.ru about love.<br>
### Web-version uses `Streamlit`.<p>

You need to predownload:<p>
Corpora - https://www.kaggle.com/bobazooba/thousands-of-questions-about-love<p>
FastText - https://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz<p>

Usage example:<p>
![image](https://github.com/Vazelisk/love_search/blob/master/saved/example.gif)

### Guide
`streamlit_app.py` launches web-service. Before it you should to run `search_engine.py` to create all needed files, it will also score engines and launch command-line search.
  
search_engine output example:<p>
![image](https://user-images.githubusercontent.com/42929213/137906245-d96a1542-4d2c-4fc5-a6eb-9ccbdc06b2ac.png)
