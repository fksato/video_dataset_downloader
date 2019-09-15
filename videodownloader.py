import os
import re
import json
import errno
import subprocess
import pandas as pd
from tqdm import tqdm
from urllib.parse import parse_qs
from urllib.request import urlopen
from socket import error as SocketError

import logging

class VideoDownloader:
	def __init__(self, time_out=100
	             , quality='480p'
	             , fps=(24,25,30)
	             , video_duration=2.0
	             , action_delay=0.5
	             , DLOAD_DIR=None
	             , rank=None
	             , offset=0
	             , segment='training'
	             , checkpoint=False
	             , checkpoint_rate=100
	             , meta_idx=0):

		self._time_out = time_out
		self.checkpoint_rate = checkpoint_rate
		self.vid_quality = quality

		self.has_succeeded = True
		self._ytube_url = 'https://www.youtube.com/get_video_info?html5=1&video_id='

		if isinstance(fps, (list, tuple)):
			self.capture_frames = (video_duration + action_delay) * max(fps)
			self.capture_fps = max(fps)
		else:
			self.capture_frames = (video_duration + action_delay) * fps
			self.capture_fps = fps

		self.capture_frames = str(self.capture_frames)
		self.capture_fps = str(self.capture_fps)

		self._fps = fps if isinstance(fps, (list, tuple)) else [fps]

		self._video_duration = video_duration
		self._delay = action_delay
		self._size_ratio = 16.0/9.0
		self._retry = 0
		self.RETRY_MAX = 10

		self.dataset_segment = segment

		self.cur_vid_title = None
		self.vid_info = None
		self.checkpoint = checkpoint

		self._meta_pkg = None
		self.meta_idx = None
		# SETUP VIDEO DIRECTORIES:
		if not DLOAD_DIR:
			raise Exception('You must specify the directory to save videos into')
		else:
			top_dir = DLOAD_DIR
			_mk_action_dir(DLOAD_DIR)
		self.HACS_VID_DIR = os.path.join(top_dir, f'{self.dataset_segment}')
		# temp_vid directories (needed for ffmpeg trim/processing: <DLOAD_DIR>/.temp
		temp_dir = os.path.join(self.HACS_VID_DIR, '.temp')
		_mk_action_dir(temp_dir)
		#
		self.rank = rank
		self._offset = offset
		if rank is None:        # single processor
			self._temp_file = os.path.join(temp_dir, 'temp.mp4')
			self._meta_temp_file = os.path.join(temp_dir, 'temp_meta.mp4')
		else:
			self.rank = rank
			self._temp_file = os.path.join(temp_dir, f'temp_{rank}.mp4')
			self._meta_temp_file = os.path.join(temp_dir, f'temp_meta_{rank}.mp4')
		self.meta_idx = meta_idx

	def __call__(self, annotations_list):
		if self.rank is not None: print(f'processing on {self.rank}\n')
		self._setup_meta(len(annotations_list))
		sucess_cnt = 0
		for anno in annotations_list:
			vid_id = anno.video_id
			unique_id = f'{anno.id}'
			start = anno.start
			end = anno.end - self._delay
			label = anno.label
			vid_dir = os.path.join(self.HACS_VID_DIR, label)
			# download, trim, save meta:
			if self.execute(unique_id, vid_id, label, start, end, vid_dir):
				self.meta_idx += 1
				sucess_cnt += 1

			if self.checkpoint and sucess_cnt % self.checkpoint_rate:
				self._checkpoint_dloaded() # meta_rank_checkpoint.pkl

		self._pkg_meta() #meta_rank.pkl

	def _setup_meta(self, records_count):
		_meta_fname = f'meta_{self.rank}.pkl' if self.rank is not None else f'meta.pkl'
		self._meta_pkg = [{"vid_index": (self._offset + index if self.rank else index)} for index in
		                  range(records_count)]

	def get_parse_url(self, vid_id):
		_url = self._ytube_url + vid_id
		# noinspection PyBroadException
		try:
			data = urlopen(
				_url,
				timeout=self._time_out
			).read().decode()
			info = parse_qs(data)
		except:
			return False

		try:
			a = info['adaptive_fmts']
			player_resp = json.loads(info['player_response'][0])
			self.cur_vid_title = player_resp['videoDetails']['title']
			self.vid_info = info['adaptive_fmts'][0].split(",")
		except KeyError:
			self.has_succeeded = False
			return False
		return True

	def dload(self, dl_url):
		# noinspection PyBroadException
		try:
			resp = urlopen(dl_url)
		except:
			return False

		length = int(resp.headers['Content-Length'])

		with open(self._temp_file, "wb+") as fh:
			for i in tqdm((1024, length, 1024), unit_scale=1024):
				try:
					buff = resp.read(i)
					fh.write(buff)
				except SocketError as e:
					if e.errno == errno.ECONNRESET:
						self._retry += 1
						logging.warning(f'Connection reset by peer\nTrying again {self._retry}/{self.RETRY_MAX}')
						if self._retry > self.RETRY_MAX:
							self._retry = 0
							logging.warning(f'Unable to download {dl_url}\nMax retry reached')
							return False
						else:
							self.dload(dl_url)
					else:
						logging.warning(f'Unable to download\nSocket error {e}')
						return False

		self._retry = 0
		return True

	def execute(self, unique_id, vid_id, label, start, stop, vid_dir):
		if not self.get_parse_url(vid_id):
			return False

		vid_title = re.sub(r"[^a-zA-Z0-9]+", ' ', self.cur_vid_title)

		vid_title = vid_title.strip()
		fname_title = f'{vid_title.replace(" ", "_")}@{start}'

		_vid_data = {}
		for info in self.vid_info:
			x = parse_qs(info)
			if 'quality_label' not in x.keys():
				continue
			quality = x['quality_label'][0]
			_vid_data[quality] = [x['url'][0], x['size'][0], float(x['fps'][0])]

		if self.vid_quality not in _vid_data.keys():
			return False

		match = re.compile(r'([0-9]{3,})x([0-9]{3,})').search(_vid_data[self.vid_quality][1])
		if match:
			x,y = map(float, match.groups()[0:2])
		else:
			return False

		if y == 0:
			return False
		elif abs((x/y) - self._size_ratio) > 0.01:
			return False

		fps = int(_vid_data[self.vid_quality][2])

		start_frame = int((start - self._delay) * fps)

		dload_url = _vid_data[self.vid_quality][0]
		if not self.dload(dload_url):
			return False

		fname = os.path.join(vid_dir, fname_title)
		fname =  f"{fname}.mp4"

		if not self.trim_vid(fps, start_frame, fname):       # capture error in trim
			return False

		exact_dur = _get_exact_duration(fname)
		exact_dur = float(exact_dur) if exact_dur else 0

		if exact_dur != (self._delay + self._video_duration):
			os.remove(fname)
			return False

		self._add_meta(unique_id, vid_id, vid_title, fps, dload_url, start, stop, x, y, exact_dur, fname, label)
		return True

	def trim_vid(self, fps, start, fname):
		cmd = ['ffmpeg', "-y",
		       "-r", str(fps),
		       "-v", "quiet",
		       "-i", self._temp_file,
		       "-vf", f"select=gte(n\\,{start}),fps=fps={self.capture_fps}",
		       "-vframes", self.capture_frames,
		       "-c:v", "libx264",
		       "-pix_fmt",  "yuv420p",
		       "-crf", "23",
		       "-f", "mp4",
		       "-movflags", "faststart",
		       self._meta_temp_file]
		subprocess.run(cmd)
		cmd = ['ffmpeg', "-y",
		       "-v", "quiet",
		       "-i", self._meta_temp_file,
		       "-map_metadata", "-1",
		       fname]
		return 1 - subprocess.run(cmd).returncode

	def _add_meta(self, unique_id, vid_id, vid_title, fps, dload_url, start, end, width, height, dur, fname, label):
		meta = self._meta_pkg[self.meta_idx]
		meta['vid_id'] = vid_id
		meta['unique_id'] = unique_id
		meta['action_label'] = label
		meta['url'] = dload_url
		meta['quality'] = self.vid_quality
		meta['title'] = vid_title
		meta['fps'] = fps
		meta['start'] = start
		meta['end'] = end
		meta['width'] = width
		meta['height'] = height
		meta['exact_duration'] = dur
		meta['file_path'] = fname

	def _pkg_meta(self):
		df_meta_data = pd.DataFrame( self._meta_pkg )
		pkl_name = f'meta_{self.rank}.pkl' if self.rank is not None else 'meta_0.pkl'

		df_meta_data.to_pickle(os.path.join(self.HACS_VID_DIR, pkl_name))

	def _checkpoint_dloaded(self):
		meta = pd.concat([self._meta_pkg['unique_id'], self._meta_pkg['file_path']], axis=1
		                 , keys=['unique_id', 'file_path'])

		pkl_name = f'.temp/meta_{self.rank}_checkpoint.pkl' \
			if self.rank is not None else '.temp/meta_0_checkpoint.pkl'

		meta.to_pickle(os.path.join(self.HACS_VID_DIR, pkl_name))


def _get_exact_duration(fname):
	cmd = ['ffprobe', '-v', 'quiet',
	       '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', fname]
	p = subprocess.Popen(cmd,
	                     stdout=subprocess.PIPE)
	stdout = p.communicate()
	return stdout[0].decode('utf-8').strip()


def _mk_action_dir(vid_dir):
	if not os.path.exists(vid_dir):
		os.makedirs(vid_dir)
	return vid_dir
