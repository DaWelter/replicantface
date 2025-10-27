from typing import NamedTuple, IO
import numpy as np
import h5py
import argparse
import tqdm
from pathlib import Path
import contextlib
from scipy.spatial.transform import Rotation
import functools
from contextlib import closing
from pprint import pprint
from functools import lru_cache
from numpy.typing import NDArray
import cv2
import scipy.io
from PIL import Image
import zipfile
from io import BytesIO
import itertools


from trackertraincode.datasets.preprocessing import depth_centered_keypoints, imread

# For segmentation mask.
COLOR_FACE = (220,57,33)
COLOR_BEARD = (118,190,70)
COLOR_CLOTHES = (135,198,199)
COLOR_BG = (0,0,0)
# The colors unfortunately came out a bit skewed. It should really be exect primary colors.


def imread(fn):
    '''Reads image.
    
    Color images are returned in RGB  format!
    '''
    img = cv2.imread(fn)
    assert img is not None, f"Failed to load image {fn}!"
    if len(img.shape)==3 and img.shape[-1]==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def depth_centered_keypoints(kpts):
    eye_corner_indices = [45, 42, 39, 36]
    center = np.average(kpts[:,eye_corner_indices], axis=1)
    kpts = np.array(kpts, copy=True)
    kpts[2] -= center[2]
    return kpts


def map_indices(full_head_points, subset_indices):
    m = np.full(np.amax(full_head_points) + 1, fill_value=-1, dtype=np.int64)
    m[full_head_points] = np.arange(len(full_head_points))
    new_subset_indices = m[subset_indices]
    assert np.all(new_subset_indices >= 0)
    return new_subset_indices


@functools.lru_cache()
def get_landmark_indices(dataset_root: Path):
    with closing(np.load(dataset_root / 'head_indices.npz')) as f:
        # Overall head. All vertices.
        head_indices = f['indices']
    with closing(np.load(dataset_root / 'landmark_indices.npz')) as f:
        # Landmarks roughly matching the 68 points scheme from 300W-LP and others.
        landmark_indices = f['indices']
    with closing(np.load(dataset_root / 'face_indices.npz')) as f:
        # Only the face area. Dense vertices. Used for creating the bounding box.
        face_indices = f['indices']
    # Map landmark indices to reduced set of head vertices.
    new_landmark_indices = map_indices(head_indices, landmark_indices)
    new_face_indices = map_indices(head_indices, face_indices)
    return new_landmark_indices, new_face_indices


def _screen_to_image(p, img_size):
    return (1.0 - p) / 2.0 * img_size

@lru_cache(maxsize=2)
def get_image(image_filename : Path):
    return imread(str(image_filename))

@lru_cache(maxsize=2)
def get_segmentation(filename : Path) -> NDArray[np.uint8]:
    '''Return RGB, uint8, HWC segmentation image.'''
    seg_array = imread(str(filename))
    assert len(seg_array.shape)==3 and seg_array.shape[-1]==3 and seg_array.dtype==np.uint8
    return seg_array


def check_valid(image_filename : Path):
    #seg_filename = image_filename.parent / (image_filename.stem + '_mask.png')

    # seg_array = imread(str(seg_filename))
    # assert len(seg_array.shape)==3 and seg_array.shape[-1]==3 and seg_array.dtype==np.uint8
    # h, w, _ = seg_array.shape
    # num_face = np.count_nonzero(np.amax(np.abs(seg_array.astype(np.int32) - np.asarray(COLOR_FACE)),axis=-1) < 50)

    # if str(seg_filename).endswith('01792_mask.png') or str(seg_filename).endswith('00000_mask.png'):
    #     print (num_face, h*w, num_face/(h*w))
    #     from matplotlib import pyplot
    #     pyplot.imshow(seg_array)
    #     pyplot.show()

    # if num_face < h*w*0.01:
    #     # Insufficient face area
    #     return False

    image_array = get_image(image_filename)
    avg_brightness = np.average(image_array)
    if avg_brightness < 20 and np.percentile(np.ravel(np.average(image_array,axis=-1)), 98) < 20:
        # Too dark and no bright areas
        return False
    
    return True


def _calc_mask_for_class(seg_array, class_colors):
    return np.amax(np.abs(seg_array.astype(np.int32) - np.asarray(class_colors)),axis=-1) < 20


def generate_roi_from_points(landmarks):
    min_ = np.amin(landmarks[...,:2], axis=-2)
    max_ = np.amax(landmarks[...,:2], axis=-2)
    roi = np.concatenate([min_, max_], axis=-1).astype(np.float32)
    return roi


def roi_intersection(roi1, roi2):
    '''x0y0x1y1 format. Shapes: (...,4)'''
    xymin = np.maximum(roi1[...,:2], roi2[...,:2])
    xymax = np.minimum(roi1[...,2:], roi2[...,2:])
    roi = np.concatenate([xymin,xymax], axis=-1)
    return roi



def generate_roi_from_seg(image_filename : Path) -> NDArray[np.float32]:
    seg_array = get_segmentation(image_filename)
    h, w, _ = seg_array.shape

    mask = _calc_mask_for_class(seg_array, COLOR_FACE)
    points = cv2.findNonZero(mask.astype(np.uint8))

    if points is None:
        print (f"Warning ROI fallback activated for {image_filename}")
        mask = ~(_calc_mask_for_class(seg_array, COLOR_CLOTHES) | _calc_mask_for_class(seg_array, COLOR_BG))
        points = cv2.findNonZero(mask.astype(np.uint8))
    
    if 0:
        from matplotlib import pyplot
        fig, ax = pyplot.subplots(1,2)
        ax[0].imshow(mask)
        ax[1].imshow(seg_array)
        pyplot.show()

    assert points.ndim==3 and points.shape[1]==1 and points.shape[2]==2
    bbox = generate_roi_from_points(points[:,0,:])

    bw, bh = (bbox[2:] - bbox[:2])
    if (bw < 32 or bh < 32) or (bw > 2*w//3 or bh > 2*h//3):
        return np.zeros((4,), dtype=np.float32)
    return bbox


class Labels(NamedTuple):
    # The coordinate system is such that x = right, y = down, z = into the screen
    rot : Rotation
    xy : NDArray[np.float64] # x, y, in [-1,1] normalized image space
    size : float # head radius in [-1,1] normalized image space
    bbox : NDArray[np.float32] # x0y0x1y1
    landmarks : NDArray[np.float64] # (68,3) - in [-1,1] normalized image space
    image_resolution : tuple[int|float,int|float] # For convenience


def convert(filename: Path):
    with contextlib.closing(np.load(filename)) as f:
        modelview = f['modelview']
        projection = f['projection']
        vertices = f['vertices']
        resolution = f['resolution']
    assert np.isclose(projection[0, 0], projection[1, 1]), "FOV should be symmetric"
    # Rotation to compensate different axis choices between blender and this project.
    rx = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    rx44 = np.eye(4)
    rx44[:3, :3] = rx
    # Position and size
    headbone_to_eye_center = np.asarray([0.0, -0.064, -0.086, 1.0])
    facepos3d = rx44.T @ modelview @ rx44 @ headbone_to_eye_center
    # TODO: Should the headsize be accurate to every individual sample? How would I calculate that?
    #       Currently the head shape doesn't vary a lot though.
    headradius3d = 0.1  # Hardcoded approximation for all heads. (meters)
    img_size = float(resolution)
    p = projection @ facepos3d
    p = p / p[3]
    depth = facepos3d[2]
    p[:2] = _screen_to_image(p[:2], img_size)
    # Weak perspective approximation for the image-space size of the head.
    # Note the 0.5 comes from the screen to image mapping because the image
    # spans the range [-1,1].
    p[2] = headradius3d * projection[0, 0] / depth * img_size * 0.5
    # Rotation
    rot = Rotation.from_matrix(rx.T @ modelview[:3, :3] @ rx)
    # Vertices and bounding box
    landmark_indices, face_indices = get_landmark_indices(filename.parent)
    vertices = np.pad(vertices, [(0, 0), (0, 1)], constant_values=1.0)
    proj_vertices = (projection @ rx44.T @ modelview) @ vertices[face_indices].T
    proj_vertices /= proj_vertices[3, :]
    proj_vertices = _screen_to_image(proj_vertices[:2], img_size).T

    assert proj_vertices.ndim==2 and proj_vertices.shape[1]==2
    bbox = generate_roi_from_points(proj_vertices)

    # Landmarks
    landmarks = vertices[landmark_indices]  # - headbone_to_eye_center
    landmarks = (rx44.T @ modelview @ landmarks.T).T
    if 1:
        # Weak perspective projection 
        landmarks = -projection[0, 0] / depth * landmarks
    else:
        landmarks = landmarks @ projection.T
        landmarks = landmarks / landmarks[:,3:]
    landmarks = _screen_to_image(landmarks[:, :3], img_size)
    landmarks = depth_centered_keypoints(landmarks.T).T
    # print (landmarks.shape)
    # print ('mproj\n',projection)
    # print ('modelview', np.linalg.det(modelview[:3,:3]))
    # print ('p', p)
    return Labels(rot, p[:2], p[2], bbox, landmarks, (resolution,resolution))


def inv_aflw_rotation_conversion(rot: Rotation):
    '''Rotation object to Euler angles for AFLW and 300W-LP data

    Returns:
        Batch x (Pitch,Yaw,Roll)
    '''
    P = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    M = P @ rot.as_matrix() @ P.T
    rot = Rotation.from_matrix(M)
    euler = rot.as_euler('XYZ')
    euler *= np.asarray([1, -1, 1])
    return euler


def move_head_center_back(scale, rot, xy):
    """Moves the 2d position from between the eyes back to where it is in 300wlp."""
    local_offset = np.array([0.0, -0.26, -0.9])
    offset = rot.apply(local_offset) * scale
    return xy - offset[:2]


class DatasetWriter300WLPLike:
    def __init__(self, filename):
        self._zf = zipfile.ZipFile(filename, "w")


    def _write_labels(self, file : str | IO, labels: Labels):
        human_head_radius_micron = 100.0e3
        w, h = labels.image_resolution
        scale = labels.size / human_head_radius_micron / w * 224.0 / 0.5
        xy = move_head_center_back(labels.size, labels.rot, labels.xy)
        tx = xy[0]
        ty = h - xy[1]
        tz = 0.0
        pitch, yaw, roll = inv_aflw_rotation_conversion(labels.rot)
        x0,y0,x1,y1 = labels.bbox
        y0 = h - y0
        y1 = h - y1
        mat_dict = {
            'Pose_Para': [[pitch, yaw, roll, tx, ty, tz, scale]],
            'pt3d_68' : (labels.landmarks * np.asarray([1.,1.,-1]) ).T,  # output shape (3,68)
            # 300W-LP and AFLW2k-3D have "roi" labels. I never figured out their meaning so here is something else.
            'roi' : [[x0,y0,x1,y1]]
        }
        scipy.io.savemat(file, mat_dict)


    def write(self, name: str, image_filename : str, sample: Labels):
        assert image_filename.endswith(".jpg")
        label_buffer = BytesIO()
        self._write_labels(label_buffer, sample)
        self._zf.writestr(f"{name}.mat", label_buffer.getvalue())
        self._zf.write(image_filename, f"{name}.jpg")


    def close(self):
        self._zf.close()


def as_hpb(rot):
    '''This uses an aeronautic-like convention. 
    
    Rotation are applied (in terms of extrinsic rotations) as follows in the given order:
    Roll - around the forward direction.
    Pitch - around the world lateral direction
    Heading - around the world vertical direction
    '''
    return rot.as_euler('YXZ')


def convert_replicantface_folder(sources : list[str], destination : str, count : int | None, parent_in_zip : Path):
    """Converst and zips the files that the blender script creates.
    
    The output format is mostly compatible with the 300wlp and aflw2000-3d datasets.
    The .mat file will have Pose_Para, pt3d_68 and roi.
    
    Beware, the roi is the facial bounding box in x0y0x1y1 format. This is different from the other datasets.
    """
    assert destination.endswith(".zip")
    assert not parent_in_zip.is_absolute()

    label_files = sorted(list(itertools.chain.from_iterable(
        Path(s).glob('face_[0-9]*.npz') for s in sources)))

    if count:
        label_files = label_files[: count]

    with closing(DatasetWriter300WLPLike(destination)) as output_ds:
        counter = 0

        for label_file in tqdm.tqdm(label_files):
            image_file = label_file.with_name(label_file.stem + '_img.jpg')
            mask_file = label_file.with_name(label_file.stem + '_mask.png')
            
            if not check_valid(image_file):
                continue

            bbox = generate_roi_from_seg(mask_file)
            
            bw, bh = bbox[2:] - bbox[:2]
            if bw <= 32 or bh <= 32:
                continue

            labels = convert(label_file)

            h,_,_ = as_hpb(labels.rot)
            if abs(h) > 0.5*np.pi:
                continue

            output_ds.write(str(parent_in_zip / label_file.stem), str(image_file), labels)

            counter += 1
        
        print("Wrote", counter, "samples")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('destination', help="Destination file", type=str)
    parser.add_argument('source', help="source directories", type=str, nargs='*')
    parser.add_argument('-n', dest='count', type=int, default=None)
    parser.add_argument('--parent', type=str, default='replicantface')
    args = parser.parse_args()
    
    convert_replicantface_folder(args.source, args.destination, args.count, parent_in_zip=Path(args.parent))