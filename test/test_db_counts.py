import pytest
from videoDB import VideoDBWorker

class TestDBCounts:
	@pytest.mark.parametrize(['db', 'segment', 'expected_pos', 'expected_neg', 'expected_all']
		, [
             ('HACS_ds', 'training', 572881, 936616, 1509497)
             , ('HACS_ds', 'validation', 13047, 7202, 20249)
	         , ('HACS_ds', 'testing', 20296, 0, 20296)
         ])
	def test_hacs_counts(self, db, segment, expected_pos, expected_neg, expected_all):
		_db_conn = VideoDBWorker(db)
		action_classes = _db_conn.get_actions_list(segment=segment)
		temp = _db_conn.get_actions_list(segment=segment, IGNORE_NEG_CLASS=False) if segment != 'testing' else []

		assert set(action_classes) == set(temp)
		assert len(action_classes) == 200 if segment != 'testing' else len(action_classes) == 0

		pos_vid_cnts = 0
		all_vid_cnts = 0

		for action in action_classes:
			pos_vid_cnts += _db_conn.get_anno_cnt(label=action, segment=segment)
			all_vid_cnts += _db_conn.get_anno_cnt(label=action, segment=segment, IGNORE_NEG_CLASS=False)

		if segment == 'testing':
			pos_vid_cnts = all_vid_cnts = _db_conn.get_anno_cnt(label=None, segment=segment)

		assert pos_vid_cnts == expected_pos
		assert all_vid_cnts - pos_vid_cnts == expected_neg
		assert all_vid_cnts == expected_all

		_db_conn.close_db()
