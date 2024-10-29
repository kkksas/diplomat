from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
def save_in_scv_format(My_Labels , Name):
    K = []
    for i in range(len(My_Labels)):
        K.append(My_Labels[i])

    np.savetxt(Name, K, delimiter = ',')

df = pd.read_excel('./spreadcheek.xlsx')
#векторизация тематик
vectorizerTheme = CountVectorizer()
theme = vectorizerTheme.fit_transform(df['тематика'])
#print(vectorizerTheme.get_feature_names_out())
#print(theme.toarray())
dfTheme= pd.DataFrame(theme.toarray(), columns=vectorizerTheme.get_feature_names_out())
print(dfTheme)
#сохраняем в csv
#np.savetxt("themeVect.csv", theme.toarray(),fmt='%d',  delimiter=",")

#векторизация активностей
vectorizerActivity = CountVectorizer()
activity = vectorizerActivity.fit_transform(df['активности'].values.astype('U'))
dfActivity= pd.DataFrame(activity.toarray(), columns=vectorizerActivity.get_feature_names_out())
print(dfActivity)
#np.savetxt("activityVect.csv", activity.toarray(),fmt='%d', delimiter=",")

#векторизация городов подумать как их соединять
vectorizerCity = CountVectorizer()
cityVect = vectorizerCity.fit_transform(df['Город'].values.astype('U'))
dfCity= pd.DataFrame(cityVect.toarray(), columns=vectorizerCity.get_feature_names_out())
print(dfCity)
#векторизация проживания если от названий избавится то будет норм
vectorizerLiving = CountVectorizer()
livingVect = vectorizerLiving.fit_transform(df['проживание'].values.astype('U'))
dfLiving= pd.DataFrame(livingVect.toarray(), columns=vectorizerLiving.get_feature_names_out())
print(dfLiving)
conDF = pd.concat([dfTheme, dfActivity, dfCity, dfLiving], axis=1, join='outer')
print(conDF)
conDF.to_csv('out.csv', header=True, sep='\t', encoding='utf-8', index=False)