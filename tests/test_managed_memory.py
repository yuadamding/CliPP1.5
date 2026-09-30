"""Resource admission is explicit and bounded; numerical code is unchanged."""
import pytest
from clipp1d.cuda.managed_memory import HOST_HEADROOM, current_capacity, host_capacity


def test_ordinary_fit_cannot_inherit_managed_capacity():
    assert current_capacity('cuda:0') is None


def test_host_capacity_is_bounded_by_reservation_availability_and_cgroup():
    gib = 1024**3
    assert host_capacity(384*gib, 900*gib) == 384*gib-HOST_HEADROOM
    assert host_capacity(384*gib, 180*gib) == 180*gib-HOST_HEADROOM
    assert host_capacity(384*gib, 900*gib, 100*gib) == 100*gib-HOST_HEADROOM


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), 20.5])
def test_invalid_host_limits_cannot_authorize_oversubscription(value):
    with pytest.raises(ValueError):
        host_capacity(value, 1024**4)


def test_host_headroom_is_never_borrowed():
    with pytest.raises(MemoryError):
        host_capacity(HOST_HEADROOM, 1024**4)
