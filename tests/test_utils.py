import pickle as pkl
import random
from unittest.mock import patch

from src.utils import *
from src.utils import _RedisWrapper


def test_overlay_segmentation_on_image():
    resource_dir = os.path.join(os.path.dirname(
        __file__), '../resources/overlay_segmentation_on_image')
    base_image = Image.open(os.path.join(resource_dir, 'base_image.png'))
    segmentation_slice = Image.open(os.path.join(
        resource_dir, 'segmentation_slice.png'))

    with open(os.path.join(resource_dir, 'header.pkl'), 'rb') as f:
        header = pkl.load(f)

    gt = Image.open(os.path.join(resource_dir, 'result.png'))

    result = overlay_segmentation_on_image(
        segmentation_slice=np.array(segmentation_slice), image=np.array(base_image), header=header, alpha=0.6)

    np.testing.assert_almost_equal(np.array(gt), np.array(result))


@patch.object(redis, 'Redis', return_value=dict())
class TestRedisWrapper:
    def test_redis_wrapper(self, *args):
        @RedisWrapper(cache_keys=['b', 'c'])
        def func(a, b, c):
            return random.randint(0, 10)

        func(a=1, b=2, c=3)

        with patch.object(_RedisWrapper, '_fetch_key') as spy:
            func(a=1, b=2, c=3)
            spy.assert_called_once()
            spy.reset_mock()

            func(a=15, b=2, c=3)
            spy.assert_called_once()
