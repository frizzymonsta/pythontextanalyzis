import pandas as pd

# Импорт очищенного от emoji датасета.
df = pd.read_csv('data/formated_data.csv', delimiter=',')

# Перевод в нижний регистр.
print('Setting up a lower case...')
df['text'] = df['text'].str.lower()

# Удаление мусора. Точки, запятые, другие спецсимволы.
print('Cleaning tabs, new lines and multiple whitespaces...')

df['text'] = df['text'].str.replace('\n', '')
df['text'] = df['text'].str.replace(',', '')
df['text'] = df['text'].str.replace('.', '')
df['text'] = df['text'].str.replace('#', '')
df['text'] = df['text'].str.replace(')', '')
df['text'] = df['text'].str.replace('(', '')
df['text'] = df['text'].str.replace('/', '')
df['text'] = df['text'].str.replace('!', '')
df['text'] = df['text'].str.replace('?', '')
df['text'] = df['text'].str.replace('-', '')
df['text'] = df['text'].str.replace('\t', '')
df['text'] = df['text'].str.replace('@', '')
df['text'] = df['text'].str.replace('—', '')
df['text'] = df['text'].str.replace('№', '')
df['text'] = df['text'].str.replace('%', '')
df['text'] = df['text'].str.replace(':', '')
df['text'] = df['text'].str.replace(';', '')
df['text'] = df['text'].str.replace('–', '')
df['text'] = df['text'].str.replace('_', '')
df['text'] = df['text'].str.replace('ღ18', '')
df['text'] = df['text'].str.replace(' {2,}', '', regex=True)
df['text'] = df['text'].str.strip()

# Удаление ссылок в разделе text.
print('Removing URLs...')
df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

# Удаление NaN строк, если они есть.
print('Dropping NaN data...')
df.dropna()

# Экспорт датасета.
df.to_csv(r'data/totallyformated.csv')