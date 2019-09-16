import os
import re
import shutil
import pickle as pk
import pandas as pd
from glob import glob

def round_robin(src_list, num_procs):
	total = len(src_list)
	ret_list = [None] * num_procs

	if total < num_procs:
		for i in range(num_procs):
			if i < total:
				ret_list[i] = [src_list[i]]
			else:
				ret_list[i] = []
		return ret_list

	per_proc = int(total/ num_procs)
	rem = total % num_procs
	start = 0
	end = 0

	for k in range(num_procs):
		end += per_proc
		ret_list[k] = src_list[start:end]
		start += per_proc

	for j in range(rem):
		ret_list[j] += [src_list[end]]
		end+=1

	return ret_list


#TODO: do not need
def action_level_round_robin(vid_db_worker, action_class, segment, num_procs, IGNORE_NEG_CLASS=True):
	action_video__list = vid_db_worker.get_anno_list(label=action_class, segment=segment
	                                                , IGNORE_NEG_CLASS=IGNORE_NEG_CLASS)

	dist_list = round_robin(action_video__list, num_procs)

	return dist_list


def ds_level_round_robin(vid_db_worker, segment, num_procs, IGNORE_NEG_CLASS=True):
	actions_list = vid_db_worker.get_actions_list(segment, IGNORE_NEG_CLASS)
	complete_dist_list = []
	for action in actions_list:
		# TODO: remove action_level
		# TODO: add check_dloaded
		temp = action_level_round_robin(vid_db_worker, segment, num_procs, IGNORE_NEG_CLASS)
		if len(complete_dist_list) == 0:
			complete_dist_list = temp
		else:
			complete_dist_list = [complete_dist_list[i] + temp[i] for i in range(len(complete_dist_list))]

	return complete_dist_list


def check_dloaded(anno_list, copy_dir, dloaded_action_meta, copy_files=False, retrieve_meta=True):
	recorded_anno = dloaded_action_meta.copy(deep=True)
	new_id = _check_id_type(recorded_anno['unique_id'].values[0])
	if new_id:
		tot_anno = {anno.id: anno for anno in anno_list}
	else:
		tot_anno = {f'{anno.video_id}{anno.start}{anno.end}': anno for anno in anno_list}

	recorded_keys = [rec_key for rec_key in dloaded_action_meta['unique_id'].values]

	for row in recorded_anno.itertuples():
		vid_src = row.file_path
		_anno = tot_anno[row.unique_id]
		vid_title = re.sub(r"[^a-zA-Z0-9]+", ' ', row.title)
		vid_title = vid_title.replace(" ", "_")
		fname = f'{vid_title}@{row.start}.mp4'
		copy_video_path = os.path.join(copy_dir, row.action_label, fname)

		recorded_anno.at[row.Index, 'file_path'] = copy_video_path
		recorded_anno.at[row.Index, 'unique_id'] = _anno.id

		if copy_files:
			if not os.path.isfile(copy_video_path):
				try:
					shutil.copy2(vid_src, copy_video_path)
				except:
					raise Exception(f'Error in shutil copy2({vid_src}, {copy_video_path})')


	dload_anno = [tot_anno[key] for key in tot_anno.keys() if key not in recorded_keys]
	if not retrieve_meta:
		return dload_anno

	return recorded_anno, dload_anno

def retreive_checkpoint_metas(chkpt_dir, get_failed=False):
	failed_chkpts = glob(os.path.join(chkpt_dir, 'meta_failed_*_checkpoint.pkl'))
	success_meta_chkpts = glob(os.path.join(chkpt_dir, '*_checkpoint.pkl'))

	if len(success_meta_chkpts) == 0:
		return

	success_meta_chkpts = list(set(success_meta_chkpts).difference(set(failed_chkpts)))

	s_meta_chkpt = []
	for i in success_meta_chkpts:
		with open(i, 'rb') as f:
			s_meta_chkpt.append(pk.load(f))

	s_meta_chkpt = pd.concat(s_meta_chkpt)

	if get_failed:
		f_meta_chkpt=[]
		for i in failed_chkpts:
			with open(i, 'rb') as f:
				f_meta_chkpt.append(pk.load(f))

		f_meta_chkpt = pd.concat(f_meta_chkpt)
		return f_meta_chkpt, s_meta_chkpt
	else:
		return s_meta_chkpt

def _check_id_type(id):
	try:
		int(id)
		return True
	except ValueError:
		return False


