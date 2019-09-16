import os
from videodownloader import VideoDownloader
from videoDB import VideoDBWorker


DLOAD_DIR = os.path.join(os.path.dirname(__file__), 'dload_dir')
a = VideoDownloader(DLOAD_DIR=DLOAD_DIR)
db_con = VideoDBWorker('HACS_ds')
anno_list = db_con.get_anno_list(label='Archery')[:5]
a(anno_list)