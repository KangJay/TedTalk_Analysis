import pandas as pd
import json
import re

data = pd.read_csv('main_janice.csv')
data8 = data.loc[data['cluster_tags_10'] == 8]
data9 = data.loc[data['cluster_tags_10'] == 9]
data = pd.concat([data8, data9])
#columnization
target = data['title', 'views', 'related_talks']
title_original_list = []
views_original_list = []
identification_list = []
hero_list = []
speaker_list = []
title_list = []
duration_list = []
slug_list = []
viewed_count_list = []
for related_talk in target[0:10]:
    related_talk = str(related_talk)
    related_talk1 = re.sub('([{])\'|([\s])\'|([:])\'', '\\1\"', related_talk)
    related_talk2 = re.sub('\'([\:])', r'"\1', related_talk1)
    related_talk3 = re.sub('\'([,])', r'"\1', related_talk2)
    related_talk_list4 = json.loads(related_talk3)
    identification, hero, speaker, title, duration, slug, viewed_count = None, None, None, None, None, None, None
    for target_dict in related_talk_list4:
        if target_dict['id']: identification = target_dict['id']
        if target_dict['hero']: hero = target_dict['hero']
        if target_dict['speaker']: speaker = target_dict['speaker']
        if target_dict['title']: title = target_dict['title']
        if target_dict['duration']: title = target_dict['duration']
        if target_dict['slug']: slug = target_dict['slug']
        if target_dict['viewed_count']: viewed_count = target_dict['viewed_count']
        
        identification_list.append(identification)
        hero_list.append(hero)
        speaker_list.append(speaker)
        title_list.append(title)
        duration_list.append(duration)
        slug_list.append(slug)
        viewed_count_list.append(viewed_count)
        
new_data = pd.DataFrame({'id': identification_list, 'hero': hero_list, 'speaker': speaker_list, 'title': title_list, 'duration': duration, 'slug': slug_list, 'viewed_count': viewed_count_list})
new_data.to_excel("output.xlsx")  
        
    
        
