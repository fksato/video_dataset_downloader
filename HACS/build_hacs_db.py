from videoDB import VideoDatasetDBBuilder

builder = VideoDatasetDBBuilder('HACS_ds')
builder.populate_db('HACS/HACS_clips_v1.1.csv')

builder.close_db()

print('db built')