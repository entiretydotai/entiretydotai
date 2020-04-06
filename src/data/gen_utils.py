import re

def clean_flair_tags(rows,):
    for j in ['<', '>']:
        rows = str(rows).replace(j, "")
        rows = re.sub(' +', ' ', str(rows))
        rows = str(rows).strip()
    return rows
