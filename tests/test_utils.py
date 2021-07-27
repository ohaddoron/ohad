import pickle as pkl

from src.utils import *


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
