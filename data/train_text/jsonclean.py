import json
import re

with open('eminem.json') as handler:
    RAW = json.load(handler)
TEXT = ''
for i in range(len(RAW)):
    TEXT += RAW[i]['song_lyrics']

TEXT = re.sub(r'([a-z])([A-Z])', r'\1\n\2', TEXT)

with open('eminem.txt', 'w', encoding='utf-8') as handler:
    handler.write(TEXT)