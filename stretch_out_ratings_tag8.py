import pandas as pd
import json
import re

tag8 = pd.read_excel('tag8.xlsx')
target = tag8[['title', 'views', 'ratings', 'comments']]

tag_name_set = []
for index, row in target.iterrows():
    print(index)
    ratings = str(row['ratings'])
    ratings1 = re.sub('\'', '\"', ratings)
    ratings2 = json.loads(ratings1)
    
    for ratings_dict in ratings2:
        if ratings_dict['name']:
            tag_name_set.append(ratings_dict['name'].strip())
    
tag_name_set = set(tag_name_set)

from collections import defaultdict
m = defaultdict(list)
print("length: ", len(tag_name_set))
for tag_name in tag_name_set: # https://stackoverflow.com/questions/6649361/creating-a-new-list-for-each-for-loop
    m[tag_name]


title_list = []
views_list = []
comments_list = []
    
for index, row in target.iterrows():
    print(index)
    title_list.append(row['title'])
    views_list.append(row['views'])
    comments_list.append(row['comments'])
    
    ratings = str(row['ratings'])
    ratings1 = re.sub('\'', '\"', ratings)
    ratings2 = json.loads(ratings1)
    
    for ratings_dict in ratings2:
        m[ratings_dict['name']].append(ratings_dict['count'])

df_dict = {'title': title_list, 'views': views_list, 'comments': comments_list}
for key, val_list in m.items():
    print("key : ", key)
    print("val_list : ", val_list)
    df_dict[key] = val_list  
      
new_data = pd.DataFrame(df_dict)
new_data.to_excel("output_tag8.xlsx")  

        
