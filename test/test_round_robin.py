import pytest
from multiproc_utils import round_robin

class TestRoundRobin:
	@pytest.mark.parametrize(['num_procs','num_elem'],[(100,1), (16,100), (16,123), (13,143)])
	def test_round_robin(self, num_procs, num_elem):
		src_list = [i for i in range(num_elem)]
		num_procs = 16

		per_proc = int(num_elem/num_procs)
		rem = num_elem % num_procs

		dist_lists = round_robin(src_list, num_procs)
		expected = [ per_proc if i >= rem != 0 else per_proc + 1 for i in range(num_procs)]
		sum_list = 0
		for idx, rr_list in enumerate(dist_lists):
			assert len(rr_list) == expected[idx]
			assert set(rr_list).intersection(dist_lists) == set(rr_list)
			sum_list += len(rr_list)

		assert sum_list == num_elem
