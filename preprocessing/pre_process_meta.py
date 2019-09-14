import os
import pickle as pk
import pandas as pd
from videodownloader import _mk_action_dir
import random
from shutil import copy
import warnings
from glob import glob

def custom_formatwarning(msg, *args, **kwargs):
	# ignore everything except the message
	return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning


class VideoProcessor():
	ACTIONCLASS_NUM_VIDS = 90   # plist size
	def __init__(self, processing_dir=None, meta_file=None):
		self.processing_dir = os.path.join(os.path.expanduser('~'), '_hacs_processing')
		if processing_dir:
			self.processing_dir = processing_dir

		self.action_sizes = {}
		_mk_action_dir(self.processing_dir)
		# self.meta_file_path = meta_file
		self.vid_meta = _load_meta(meta_file)

	def check(self, action):
		print(self.vid_meta.loc[self.vid_meta['action_label'] == action].shape[0])

	def create_lookup_meta(self, _actions=None):
		# get unique action labels in meta:
		if not _actions:
			actions_list = self.vid_meta['action_label'].unique()
		else:
			if isinstance(_actions, str):
				_actions = [_actions]
			actions_list = _actions

		for action in actions_list:
			# create processing dir:
			action_dir = os.path.join(self.processing_dir, action)
			_mk_action_dir(action_dir)
			action_lookup_file = os.path.join(action_dir,f'{action}.pkl')
			# check if metafile exists in dir:
			if not os.path.isfile(action_lookup_file):
				max_vids = self.action_sizes[action] = self.vid_meta.loc[self.vid_meta['action_label'] == action].shape[0]
				if max_vids < 200:
					warnings.warn(f'{action} only has {max_vids} available\n')
				# create look up meta:
				#   choose 90 random videos from action class from vid_meta:
				look_up_meta = pd.DataFrame()
				indices = random.sample(range(0, (max_vids - 1)), VideoProcessor.ACTIONCLASS_NUM_VIDS)

				self._get_videos(look_up_meta, action, indices, action_lookup_file)
				self._make_plist(action)

				_make_empty_review_plist(os.path.join(self.processing_dir,action, f'review_plist_{action}.m3u8'))

	def review_plist(self, action):
		if isinstance(action, str):
			action = [action]

		for _action in action:
			_review_plist = os.path.join(self.processing_dir, _action, f'review_plist_{_action}.m3u8')
			vid_list = read_plist(_review_plist)
			refresh = len(read_plist(os.path.join(self.processing_dir, _action, f'{_action}.m3u8')))>0

			if vid_list is None and not refresh:
				return

			_action_dir = os.path.join(self.processing_dir, _action)
			_lokup_meta_fname = os.path.join(_action_dir, f'{_action}.pkl')
			self.action_sizes[_action] = self.vid_meta.loc[self.vid_meta['action_label'] == _action].shape[0]

			lookup_meta = _load_meta(_lokup_meta_fname)

			# record the vids to keep
			for i in vid_list:
				i_path = os.path.join(self.processing_dir, f'{_action}/{i}')
				idx = lookup_meta.loc[lookup_meta['local_path'] == i_path].index.values

				# only consider vids that need to be reviewed
				if not lookup_meta.iloc[idx]['review'].values:
					continue
				# flag reviewed videos as approved
				lookup_meta.at[idx, 'approved'] = True
				lookup_meta.at[idx, 'remove'] = False
				lookup_meta.at[idx, 'review'] = False

			for meta_iter in lookup_meta.itertuples():
				# if review tag is false (not up for review)
				# or approved tag is true (approved)
				# or has already been removed
				# do not process: => only 'review'
				# if not meta_iter.review or meta_iter.approved or meta_iter.remove:
				# 	continue
				if meta_iter.review:
					lookup_meta.at[meta_iter.Index, 'review'] = False
					lookup_meta.at[meta_iter.Index, 'remove'] = True
					warnings.warn(f'REMOVING {meta_iter.local_path}')
					try:
						os.remove(meta_iter.local_path)
					except FileNotFoundError:
						warnings.warn(f'\t{meta_iter.local_path} is already removed\n', Warning)
						pass
			#
			lookup_meta.to_pickle(_lokup_meta_fname)
			# get the remaining number of vids needed to complete the plist
			num_new_vids = VideoProcessor.ACTIONCLASS_NUM_VIDS - lookup_meta.loc[lookup_meta['approved'] == True].shape[0]
			# if 0: done
			if num_new_vids == 0:
				print(f'{_action} has a complete playlist')
				continue
			else:
				# get more (index) videos, recreate plist for review:
				if num_new_vids <= 3:
					indices = self._get_rand_idx(_action, lookup_meta, self.action_sizes[_action] - lookup_meta.shape[0])
				else:
					indices = self._get_rand_idx(_action, lookup_meta, num_new_vids)
				if len(indices) > 0:
					self._get_videos(lookup_meta, _action, indices, _lokup_meta_fname)
					self._make_plist(_action)
				else:
					warnings.warn(f'No more videos available in meta for action {_action}\n'
								  f':::{_action} requires {num_new_vids} more video(s)', Warning)
					return

			# clear review list:
			_make_empty_review_plist(_review_plist)

	def reset_action_plist(self, _action):
		if isinstance(_action, list):
			warnings.warn('action to be reset cannot be a list\n', Warning)
			return
		if _action not in self.action_sizes:
			self.action_sizes[_action] = self.vid_meta.loc[self.vid_meta['action_label'] == _action].shape[0]

		warnings.warn(f'Meta for {_action} will be reset, approved videos will not be reset\n', Warning)

		_vid_dir = os.path.join(self.processing_dir, _action, 'vid')
		_vid_list = glob(os.path.join(_vid_dir,'*.mp4'))
		_lokup_meta_fname = os.path.join(self.processing_dir, _action, f'{_action}.pkl')
		#
		lookup_meta = _load_meta(_lokup_meta_fname)
		approved_vids = []
		for _meta in lookup_meta.itertuples():
			if _meta.local_path in _vid_list and _meta.approved:
				approved_vids.append(_meta.local_path)
				continue
			try:
				os.remove(_meta.local_path)
			except FileNotFoundError:
				warnings.warn(f'\t{_meta.local_path} is already removed\n', Warning)
				pass
			lookup_meta = lookup_meta.drop(_meta.Index)

		assert lookup_meta.shape[0] == len(approved_vids)
		assert set(lookup_meta['local_path'].values) == set(approved_vids)

		lookup_meta.to_pickle(_lokup_meta_fname)
		indices = self._get_rand_idx(_action, lookup_meta, VideoProcessor.ACTIONCLASS_NUM_VIDS - len(approved_vids))
		self._get_videos(lookup_meta, _action, indices, _lokup_meta_fname)
		self._make_plist(_action)
		_review_plist = os.path.join(self.processing_dir, _action, f'review_plist_{_action}.m3u8')
		_make_empty_review_plist(_review_plist)

	def _make_plist(self, action):
		action_dir = os.path.join(self.processing_dir, action)
		action_vid_dir = _mk_action_dir(os.path.join(action_dir, 'vid'))
		action_plist = os.path.join(action_dir, f'{action}.m3u8')
		meta = _load_meta(os.path.join(action_dir, f'{action}.pkl'))
		# for items in meta, copy vid into vid folder, and add to m3u8 file:
		m3u8 = ['#EXTM3U\n']
		for vid_info in meta.itertuples():
			# if vid_info.remove or vid_info.remove:
			# 	continue
			if vid_info.review:
				copy_file = vid_info.file_path
				copy_file = copy_file.replace('braintree/home/fksato', 'home/deer_meat/mnt/braintree')
				vid_fname = os.path.basename(copy_file)
				fname = os.path.join(action_vid_dir, vid_fname)
				copy(copy_file, fname)
				m3u8.append(f'#EXTINF:2,{vid_fname}\n')
				m3u8.append(f'vid/{vid_fname}\n')
		with open(action_plist, 'w') as f:
			f.writelines(m3u8)

	def _get_rand_idx(self, action, meta, num_new_vids):
		_used_idx = meta['lookup'].values
		max_size = self.action_sizes[action]
		indices = []

		if _used_idx.size == max_size:
			return indices

		_temp = [i for i in range(max_size) if i not in _used_idx]

		if num_new_vids >= len(_temp):
			return _temp

		for i in range(num_new_vids):
			rand_indx = random.randint(0, len(_temp) - 1)
			indices.append(_temp[rand_indx])
			del _temp[rand_indx]

		return indices

	def _get_videos(self, meta, action, indices, save_dir):
		action_meta = self.vid_meta.loc[self.vid_meta['action_label'] == action]
		f_path = os.path.join(self.processing_dir, action, 'vid')
		update_action_meta = False
		_temp_meta = pd.DataFrame()
		_lookup = []
		_review = []
		_approved = []
		_remove = []
		_local_path = []
		for i in indices:
			vid_fname = os.path.basename(action_meta.iloc[i]['file_path'])
			_temp_meta = _temp_meta.append(action_meta.iloc[i], ignore_index=True, sort=False)
			_lookup.append(i)
			_review.append(True)
			_approved.append(False)
			_remove.append(False)
			_local_path.append(os.path.join(f_path, vid_fname))

		_temp_meta = _temp_meta.reset_index()
		_temp_meta['lookup'] = _lookup
		_temp_meta['review'] = _review
		_temp_meta['approved'] = _approved
		_temp_meta['remove'] = _remove
		_temp_meta['local_path'] = _local_path

		meta = meta.append(_temp_meta, ignore_index=True, sort=False)
		if update_action_meta:
			action_meta.to_pickle(self.meta_file_path)
		# save look-up:
		meta.to_pickle(save_dir)

def _load_meta(fname):
	with open(fname, 'rb') as f:
		meta = pk.load(f)
	return meta.dropna()

def read_plist(m3u8_file):
	with open(m3u8_file, 'r') as f:
		first_line = f.readline().strip()
		if first_line != '#EXTM3U' and first_line != 'delete':
			return None
		# read lines:
		list_vid = [x.strip() for x in f.readlines() if x.startswith('vid/')]
	return list_vid

def _make_empty_review_plist(fname):
	with open(fname, 'w') as f:
		f.write('')


if __name__=='__main__':

	processing_dir = '/temp/test/vid_proc/temp'
	meta_file = '/mnt/braintree/temp/Shaving/_Check_global_meta.pkl'
	action_review = 'Shaving'
	test = VideoProcessor(processing_dir=processing_dir
	                      , meta_file=meta_file)

	test.review_plist(action_review)
