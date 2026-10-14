from vllm.utils.math_utils import round_up

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_lcm_alignment_for_k5_tp8():
    """DSA-CP pads token counts up to TP before graph dispatch; spec-decode
    buckets are rounded to 1+K.  Buckets must be lcm-aligned or a padded
    count lands between buckets and pads up to a mismatched num_reqs
    (372 -> 376 -> bucket 384: 62 rows vs 64 slots)."""
    f = NPUModelRunner._round_capture_sizes_for_tp_padding
    sizes = [
        12, 18, 24, 36, 42, 48, 60, 66, 72, 84, 90, 96, 108, 114, 120,
        132, 138, 144, 156, 162, 168, 180, 186, 192, 204, 210, 216, 228,
        234, 240, 252, 258, 264, 276, 282, 288, 300, 306, 312, 324, 336,
        354, 372, 384,
    ]
    r = f(sizes, 384, 6, 8)
    assert r is not None and all(s % 24 == 0 for s in r)
    assert all(round_up(s, 8) in set(r) for s in r)
    assert 372 not in r and 360 in r and 384 in r


def test_noop_without_spec_or_tp():
    f = NPUModelRunner._round_capture_sizes_for_tp_padding
    assert f([8, 16, 24], 24, 1, 8) is None
    assert f([8, 16, 24], 24, 6, 1) is None
    assert f([], 24, 6, 8) is None


def test_query_len_dividing_tp_still_explicit():
    # lcm(4,8)=8 > q=4: helper still returns the aligned list (multiples of 8)
    r = NPUModelRunner._round_capture_sizes_for_tp_padding(
        [8, 12, 16, 20, 24], 24, 4, 8
    )
    assert r == [8, 16, 24]
