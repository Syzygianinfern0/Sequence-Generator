import re

ACCEPTED = r'[a-zA-Z\s,.:;\'\"!“”’-]'
SPACES = r'\s{2,}'
CLEAN = ''

with open('mkb.txt', 'rb') as handler:
    TEXT = handler.read().decode('utf-8')
    # print(TEXT[:250])
    for c in TEXT:
        if not bool(re.match(ACCEPTED, c)):
            c = ''
        CLEAN += c

    CLEAN = re.sub(SPACES, ' ', CLEAN)
    # for c in CLEAN:
    #     if bool(re.match(SPACES, c)):
    #         c = ''
    #     SUPERCLEAN += c

with open('mkb-clean.txt', 'w', encoding='utf-8') as handler:
    handler.write(CLEAN)

# TEXT = r'ि न जात”  associated wit'
# for c in TEXT:
#     print(f"{c} : {bool(re.match(ACCEPTED, c))}")
