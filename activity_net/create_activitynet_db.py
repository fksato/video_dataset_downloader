import os
from videoDB import VideoDatasetDBBuilder, VideoDBWorker


test_action = 'Archery'
import csv
a_test = f'{os.path.dirname(__file__)}/../HACS/HACS_clips_v1.1.csv'
act_cnt = 0
dup={}
duplicate=[]
with open(a_test, 'r') as csvf:
	data = csv.reader(csvf)
	line = 0
	for row in data:
		if line == 0:
			line += 1
			continue
		if row[5] == -1:
			continue

		video_class = row[0]
		vid_sub = row[2]
		vid_sample = row[5]
		if video_class != test_action or vid_sub in ['testing', 'training'] or vid_sample=="-1":
			continue
		video_ID = row[1]
		# vid_sub = row[2]
		vid_start = row[3]
		vid_end = row[4]

		vid_key = hash(f'{video_class}{video_ID}{vid_sub}{vid_sample}{vid_start}{vid_end}')

		try:
			check = dup[vid_key]
			temp = row
			duplicate.append(temp)
			continue
		except KeyError:
			dup[vid_key] = None

		act_cnt += 1

# db_builder = VideoDatasetDBBuilder('activityNet')
# db_builder.populate_db(f'{os.path.dirname(__file__)}/activity_net.v1-3.min.json')
# db_builder.close_db()

# import json
# a_test = f'{os.path.dirname(__file__)}/activity_net.v1-3.min.json'
# act_cnt = 0
# dup={}
# duplicate=[]
# with open(a_test, 'r') as jf:
# 	jFile = json.load(jf)
# 	vids = jFile['database']
# 	for vid in vids:
# 		vid_sub = vids[vid]['subset']
# 		vid_sample = 1
# 		if vid_sub in ['testing', 'validation']:
# 			continue
# 		for i in vids[vid]['annotations']:
# 			if i['label'] == 'Applying sunscreen':
# 				vid_start = i['segment'][0]
# 				vid_end = i['segment'][1]
# 				vid_key = hash(f'Applying sunscreen{vid}{vid_sub}{vid_sample}{vid_start}{vid_end}')
# 				try:
# 					check = dup[vid_key]
# 					temp = i
# 					temp['vid_id'] = vid
# 					duplicate.append(temp)
# 				except KeyError:
# 					dup[vid_key]=None
#
# 				act_cnt += 1

print(act_cnt)

db_query = VideoDBWorker('HACS_ds')
anno_list = db_query.get_anno_list(test_action, segment='validation')

print(len(anno_list))