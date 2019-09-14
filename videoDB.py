import os
import warnings
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, Float, String, create_engine, ForeignKey
from sqlalchemy.ext.declarative import as_declarative, declared_attr


@as_declarative()
class Base(object):
	@declared_attr
	def __tablename__(cls):
		return cls.__name__.lower()
	id = Column(Integer, primary_key=True)


class Annotation(Base):
	video_id = Column(String)
	label = Column(String)
	start = Column(Float)
	end = Column(Float)
	subset = Column(String)
	sample = Column(Integer)


class Taxonomy(Base):
	id = Column(Integer, primary_key=False)
	nodeId = Column(Integer, primary_key=True)
	nodeName = Column(String, ForeignKey('annotation.label'))
	parentName = Column(String)
	parentId = Column(Integer)


class VideoDBWorker:
	_db=None
	_engine=None

	def __init__(self, db_name):
		if not os.path.isfile(f'{os.path.dirname(__file__)}/video_DB/{db_name}.db'):
			raise Exception(f'DB {db_name} does not exist')
		else:
			self.__class__._retrieve_DB(db_name)

	@property
	def _getDB(self):
		return self.__class__._db

	@classmethod
	def _retrieve_DB(cls, db_name):
		if cls._db is not None:
			warnings.warn(f'Already connected to a DB\nPlease disconnect first')
			return
		else:
			cls._engine = create_engine(f"sqlite:///{os.path.dirname(__file__)}/video_DB/{db_name}.db", echo=False)
			Annotation.metadata.create_all(cls._engine)
			Taxonomy.metadata.create_all(cls._engine)
			cls._db = sessionmaker(bind=cls._engine)()


	def get_anno_list(self, label, segment='training', IGNORE_NEG_CLASS=True):
		if not IGNORE_NEG_CLASS:
			return self._getDB.query(Annotation).filter(Annotation.label == label, Annotation.subset == segment).all()
		return self._getDB.query(Annotation).filter(Annotation.label == label, Annotation.subset == segment,
		                                    Annotation.sample == 1).all()

	def get_actions_list(self, segment='training', IGNORE_NEG_CLASS=True):
		# [temp[0] for temp in act_db.query(Taxonomy.parentName).distinct().all()]
		if not IGNORE_NEG_CLASS:
			return [temp[0] for temp in self._getDB.query(Annotation.label).filter(Annotation.subset == segment).distinct().all()]
		return [temp[0] for temp in self._getDB.query(Annotation.label).filter(Annotation.subset == segment
		                                                  , Annotation.sample == 1).distinct().all()]

	@classmethod
	def close_db(cls):
		cls._db = None
		cls._engine.dispose()


class VideoDatasetDBBuilder:
	_db = None
	_engine = None

	def __init__(self, db_name):
		self.__class__._createDB(db_name)

	@classmethod
	def close_db(cls):
		cls._db = None
		cls._engine.dispose()

	@property
	def db(self):
		return self.__class__._db

	@classmethod
	def _createDB(cls, db_name):
		if cls._db:
			warnings.warn(f'Connecting to {db_name}')
			cls.close_db()
		cls._engine = create_engine(f"sqlite:///{os.path.dirname(__file__)}/video_DB/{db_name}.db", echo=False)
		Annotation.metadata.create_all(cls._engine)
		Taxonomy.metadata.create_all(cls._engine)
		cls._db = sessionmaker(bind=cls._engine)()

	def populate_db(self, file_name):
		ext = os.path.splitext(file_name)[-1].lower()
		if ext == ".csv":
			self._populate_csv(file_name)
		elif ext == '.json':
			self._populate_json(file_name)
		else:
			raise Exception(f'cannot read: {ext} file extension')

	def _populate_csv(self, video_data_csv):
		if not self.db:
			raise Exception('DB could not be loaded/created')

		import csv

		with open(video_data_csv) as csv_data:
			csv_reader = csv.reader(csv_data)

			line = 0
			testing_cnt = 0
			for row in csv_reader:
				if line == 0:
					line += 1
					continue
				if row[5] == -1:
					continue

				video_class = row[0]
				if video_class == '':
					video_class = testing_cnt
					testing_cnt += 1

				video_ID = row[1]
				vid_sub = row[2]
				vid_start = row[3]
				vid_end = row[4]
				vid_sample = row[5]

				vid_key = hash(f'{video_class}{video_ID}{vid_sub}{vid_sample}{vid_start}{vid_end}')

				anno = Annotation(id=vid_key, video_id=video_ID, label=video_class, start=vid_start, end=vid_end
								  , subset=vid_sub, sample=vid_sample)

				self.db.add(anno)

			self.db.commit()

	def _populate_json(self, video_data_json):
		if not self.db:
			raise Exception('DB could not be loaded/created')

		import json

		with open(video_data_json, 'r') as f:
			jFile = json.load(f)

			if not self.db.query(Taxonomy).first():
				tax = jFile['taxonomy']
				for idx, _tax in enumerate(tax):
					_tax['id'] = idx
					self.db.add(Taxonomy(**_tax))

			activity = jFile['database']
			testing_cnt = 0
			for _act in activity.keys():
				info = activity[_act]
				vid_sub = info['subset']
				vid_sample = 1

				if vid_sub == 'testing':
					vid_start = 0
					vid_end = info['duration']
					label = testing_cnt
					vid_key = hash(f'{label}{_act}{vid_sub}{vid_sample}{vid_start}{vid_end}')

					anno = Annotation(id=vid_key, video_id=_act, label=label, start=vid_start, end=vid_end
					                  , subset=vid_sub, sample=vid_sample)
					self.db.add(anno)
					testing_cnt += 1
					continue

				for _anno in info['annotations']:
					vid_start = _anno['segment'][0]
					vid_end = _anno['segment'][1]
					label = _anno['label']
					vid_key = hash(f'{label}{_act}{vid_sub}{vid_sample}{vid_start}{vid_end}')

					anno = Annotation(id=vid_key, video_id=_act, label=label, start=vid_start, end=vid_end
					                  , subset=vid_sub, sample=vid_sample)
					self.db.add(anno)
			self.db.commit()



if __name__=='__main__':
	test = VideoDatasetDBBuilder('testing')
	test.populate_db('HACS/HACS_clips_v1.1.csv')

	test.close_db()

	# test = VideoDBWorker('HACS_ds')
	# a = test.get_anno_list('Fun sliding down')

	# print(a)
