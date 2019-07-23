import re

with open('movie_lines.txt', 'r') as reader:
    TEXT = reader.read()

CLEAN = ''

for c in TEXT:
    if c.isalpha() or c == '\n' or c == ' ':
        CLEAN += c

re.sub(r'\s{2,}', '', CLEAN)

with open('predicter.txt', 'w') as writer:
    writer.write(CLEAN)
