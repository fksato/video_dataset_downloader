import os
import shutil

def round_robin(src_list, num_procs):
	total = len(src_list)
	per_proc = int( total/ num_procs)
	rem = total % num_procs
	ret_list = [None] * num_procs
	start = 0
	end = per_proc

	for i in range(num_procs):
		ret_list[i] = src_list[start:end]
		start += per_proc
		end += per_proc

	rem_distribute = 0
	while rem_distribute < rem:
		idx = rem_distribute % num_procs
		ret_list[idx].append(src_list[rem_distribute+start])
		rem_distribute += 1

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


def check_dloaded(anno_list, copy_dir, dloaded_action_meta, move_files=False, retrieve_meta=True):
	tot_anno = {f'{anno.video_id}{anno.start}{anno.end}': anno for anno in anno_list}
	recorded_anno = dloaded_action_meta[dloaded_action_meta['unique_id'].isin(tot_anno.keys())]
	dload_anno=[]

	for row in recorded_anno.itertuples():
		vid_src = row.file_path
		_anno = tot_anno[row.unique_id]
		dload_anno.append(_anno)

		vid_title = row.title
		vid_title = vid_title.replace(" ", "_")
		fname = f'{vid_title}@{row.start}.mp4'
		copy_video_path = os.path.join(copy_dir, row.action_label, fname)

		recorded_anno.at[row.Index, 'file_path'] = copy_video_path
		recorded_anno.at[row.Index, 'unique_id'] = _anno.id

		if move_files:
			if not os.path.isfile(copy_video_path):
				try:
					shutil.copy2(vid_src, copy_video_path)
				except:
					raise Exception(f'Error in shutil copy2({vid_src}, {copy_video_path})')

	if not retrieve_meta:
		return dload_anno

	return recorded_anno, dload_anno

