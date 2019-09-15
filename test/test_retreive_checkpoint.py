import pytest

class TestRetreiveCheckpoint:
	@pytest.mark.parametrize(['vid_dir', 'segment', 'expected_shape']
							, [
		                            ('/braintree/home/fksato/dev/video_dataset_dloader/test/dload_dir'
		                                , 'training', (1546, 14) )
							   ])
	def test_retreive_checkpoint(self, vid_dir, segment, expected_shape):
		import os
		from multiproc_utils import retreive_checkpoint_metas

		chkpt_dir = os.path.join(vid_dir, segment, '.temp')
		chk_pts = retreive_checkpoint_metas(chkpt_dir, get_failed=False)

		assert chk_pts.shape == expected_shape