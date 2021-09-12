import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Выбор цветовой гаммы графиков.
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# Импорт очищенного набора данных.
df = pd.read_csv('data/totallyformated.csv')

# Распределение записей в наборе данных по ярлыкам.
df['label'].value_counts().plot(kind='bar', title='Распределение записей в наборе данных по ярлыкам.');
plt.show()

# Распределение записей в наборе данных по id пользователей.
df['id'].value_counts().plot(kind='bar', title='Распределение записей в наборе данных по id пользователей.');
plt.show()

