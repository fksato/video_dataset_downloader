import os
import pickle as pk
import pandas as pd
from videoDB import VideoDBWorker
from multiproc_utils import round_robin, check_dloaded

import pytest


class TestHACSDload:
	@pytest.mark.parametrize(['db_name', 'segment', 'ignore_reg', 'expected_tot'], [('HACS_ds', 'training', True, 572881)])
	def test_parallel_process(self, db_name, segment, ignore_reg, expected_tot):
		# max cpu processors:
		num_proc = 16
		HACS_VID_DIR = '/braintree/data2/active/common/HACS'
		meta_path = '/braintree/home/fksato/dev/video_dataset_dloader/dloaded/dloaded.pkl'
		#
		copy_dir = os.path.join(HACS_VID_DIR, segment)
		hacs_db = VideoDBWorker(db_name)
		actions_list = hacs_db.get_actions_list()

		complete_dist = []
		previously_dloaded = []
		tot_vid_cnt = 0

		with open(meta_path, 'rb') as f:
			global_dloaded_meta = pk.load(f)

		for action in actions_list:
			if not os.path.exists(os.path.join(HACS_VID_DIR, segment, action)):
				os.makedirs(os.path.join(HACS_VID_DIR, segment, action))

			anno_list = hacs_db.get_anno_list(label=action, segment=segment, IGNORE_NEG_CLASS=ignore_reg)
			dloaded_action_meta = global_dloaded_meta[global_dloaded_meta['action_label'] == action]
			recorded_df, dist_list = check_dloaded(anno_list, copy_dir, dloaded_action_meta, copy_files=False
			                                       , retrieve_meta=True)
			previously_dloaded.append(recorded_df)
			tot_vid_cnt += len(dist_list)

			dist_list = round_robin(dist_list, num_proc)
			if len(complete_dist) == 0:
				complete_dist = dist_list
			else:
				complete_dist = [complete_dist[i] + dist_list[i] for i in range(len(complete_dist))]

		# complete_dist
		previously_dloaded = pd.concat(previously_dloaded, ignore_index=True)
		previously_dloaded = previously_dloaded.dropna()

		assert previously_dloaded.shape[0] + tot_vid_cnt == expected_tot

		unique_vid_cnt = len(set.union(*[set(i) for i in complete_dist]))
		assert unique_vid_cnt == tot_vid_cnt
