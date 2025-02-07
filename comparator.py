import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


questions= pd.read_csv('./vectorized_questions.csv')
opis = pd.read_csv('./vectorized_opis.csv')


print("вопрос", questions)
print("описание", opis)

# Размерность векторов
vector_size = 1024
# Количество векторов
number_of_entries = opis.shape[0]

similarity = []
for i in range(number_of_entries):
    c = cosine(opis.iloc[i], questions.iloc[4])
    similarity.append([i, c])
sim = pd.DataFrame(similarity, columns=['i','c'])

sim =sim.sort_values(by=['c'])
print(sim)
#print(f"описание {similarity[0]} расстояние {similarity[1]}")
