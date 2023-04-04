import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].str.lower()
data_test['LocationNormalized'] = data_test['LocationNormalized'].str.lower()
data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True,inplace=True)
data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True,inplace=True)
vectorizer = TfidfVectorizer(min_df=5)
fullDescription_train=vectorizer.fit_transform(data_train['FullDescription'])#сбор статистики и трансформация характетистик в числа
fullDescription_test=vectorizer.transform(data_test['FullDescription'])#трансформация на основе ранее собранной статистики
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
#сливаем две колонки из датасета панды в словарь
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
joinedMatrix_train=hstack([fullDescription_train,X_train_categ])
clf = Ridge(alpha=1.0,random_state=241)
clf.fit(joinedMatrix_train, data_train['SalaryNormalized'])
joinedMatrix_test=hstack([fullDescription_test,X_test_categ])
predict=clf.predict(joinedMatrix_test)

print('Прогноз по зарплате')
print (' '.join(map(lambda x: str(round(x,2)), predict)))
