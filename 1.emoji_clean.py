import emoji

# Импорт предварительно размеченного в exel датасета.
with open('data/emoji_clean.txt', encoding='utf-8') as input:
    text = input.readlines()

# Очистка от emoji.
text = emoji.get_emoji_regexp().sub("", str(text))

# Сохранение датасета в data/emoji_cleaned.txt.
with open('data/emoji_cleaned.txt', 'w', encoding='utf-8') as file:
    file.writelines(text)
