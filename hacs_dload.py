import os
import pickle as pk
import pandas as pd
from glob import glob
import multiprocessing as mp
from videoDB import VideoDBWorker
from videodownloader import VideoDownloader
from multiproc_utils import round_robin, check_dloaded

HACS_TOTAL_TRAINING = 572881
def call_func(obj, anno_data):
	return obj(anno_data)

if __name__ == "__main__":
	# max cpu processors:
	num_procs = 16
	HACS_VID_DIR = '/braintree/data2/active/common/HACS'
	#
	print(HACS_VID_DIR)
	#
	meta_path = '/braintree/home/fksato/dev/video_dataset_dloader/dloaded/dloaded.pkl'
	#
	segment = 'training'  # 'training' 'validation'
	ignore_neg = True
	db_name = 'HACS_ds'
	copy_dir = os.path.join(HACS_VID_DIR, segment)
	hacs_db = VideoDBWorker(db_name)
	actions_list = hacs_db.get_actions_list()

	complete_dist = []

	PROD = True
	previously_dloaded = []
	tot_vid_cnt = 0

	with open(meta_path, 'rb') as f:
		global_dloaded_meta = pk.load(f)

	for action in actions_list:
		if not os.path.exists(os.path.join(HACS_VID_DIR, segment, action)):
			os.makedirs(os.path.join(HACS_VID_DIR, segment, action))

		anno_list = hacs_db.get_anno_list(label=action, segment=segment, IGNORE_NEG_CLASS=ignore_neg)
		dloaded_action_meta = global_dloaded_meta[global_dloaded_meta['action_label'] == action]
		recorded_df, dist_list = check_dloaded(anno_list, copy_dir, dloaded_action_meta, copy_files=PROD
		                                       , retrieve_meta=True)
		previously_dloaded.append(recorded_df)
		tot_vid_cnt += len(dist_list)
		dist_list = round_robin(dist_list, num_procs)
		if len(complete_dist) == 0:
			complete_dist = dist_list
		else:
			complete_dist = [complete_dist[i] + dist_list[i] for i in range(len(complete_dist))]

	# complete_dist
	previously_dloaded = pd.concat(previously_dloaded, ignore_index=True)

	assert tot_vid_cnt+previously_dloaded.shape[0]==HACS_TOTAL_TRAINING

	previously_dloaded.to_pickle(os.path.join(HACS_VID_DIR, f'{segment}/previously_dloaded.pkl'))

	print(f'Total video count for download: {tot_vid_cnt}')

	offsets = [0] + [len(x) for x in complete_dist[:-1]]
	assert len(offsets) == num_procs

	pool = mp.Pool(num_procs)

	dloaders = []
	for proc in range(num_procs):
		hacs_dloader = VideoDownloader(DLOAD_DIR=HACS_VID_DIR
					, rank=proc
					, offset=offsets[proc]
					, segment='training'
					, checkpoint=True
					, checkpoint_rate=100
					, meta_idx=0)
		dloaders.append(hacs_dloader)

	dist_args = zip(dloaders, complete_dist)

	processes = pool.starmap(call_func, [arg for arg in dist_args])
	pool.close()
	pool.join()

	#merge all aviailable pkl meta files into global meta file:
	combined_fname = glob(os.path.join(HACS_VID_DIR, f'{segment}/global_meta.pkl'))

	if combined_fname:
		with open(combined_fname,'rb') as f:
			combined_meta = pk.load(f)
	else:
		combined_meta = pd.DataFrame()

	# global_entry = 0
	for i in range(num_procs):
		#read rank meta files:
		with open(os.path.join(HACS_VID_DIR, f'{segment}/meta_{i}.pkl'), 'rb') as f:
			rank_meta = pk.load(f)
			rank_meta = rank_meta.dropna()
			combined_meta = combined_meta.append(rank_meta, ignore_index=True)

	# save global_meta:
	combined_meta.to_pickle(os.path.join(HACS_VID_DIR, f'{segment}/global_meta.pkl'))
