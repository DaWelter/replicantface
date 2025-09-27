# Replicantface dataset

This dataset has the synthetic faces with pose annotations and full 3d landmarks.
As the format is like AFLW2k-3D / 300W-LP, it has the following characteristics:

* 640 x 640 pixel jpg images
* `.mat` label files, containing
* `'Pose_Para': [[pitch, yaw, roll, tx, ty, tz, scale]]`
* `pt3d_68`: 3 x 64 point landmarks array
* `roi`: Bounding box in x0,y0,x1,y1 format. Here it is straight forward the bounding box in the same coordinate
    space as `tx` and `ty`. (In AFLW2k-3D / 300W-LP this is not the case.)

The segmentation masks are omitted.

## Benchmarking results on AFLW2k-3D

Mixing in 8% Replicantface into other training data.

|  Variant                  |  Pitch°     |    Yaw°     |      Roll°     |     Mean°     |     Geodesic°    |
| ------------------------- | ----------- | ----------- | -------------- | ------------- | ---------------- |
| Baseline                  | 4.15        | 2.69        |  2.51          |   3.12        | 5.15             |
| w/ Replicantface          | 4.13        | 2.69        |  2.51          | **3.11**      |**5.11**          |
| wo/ LMK                   | 4.16        | 2.70        |  2.53          |   3.13        | 5.15             |
| wo/ LMK, w/ Replicantface | 4.14        | 2.69        |  2.51          | **3.11**      |**5.12**          |

By default, I use auxiliary landmark predictions (to be able to use FaceSynthetics). For the results labeled "wo /LMK"
it was disabled.

The benefits are small. However, there might be benefits which are not captured in the benchmark, e.g. better generalization to glasses and pronounced facial expressions. Need more analysis. 

## Details

The bounding boxes are created from the skin segmentation. This gave better results for my models than
using the facial mesh.

Regarding diversity there are different skin tones and textures and face shapes reasonably uniformly sampled (I think).
There are around 30 accessories like glasses, hats, masks, helmets and headphones,
90 backgrounds, around 10 hairstyles, and different facial expressions. Eye movements are also present.

The distribution of poses is shown in the following plots. Blue is Replicantface and red is FaceSynthetics. To get the pose
distribution for FaceSynthetics I used a pose estimation network. They are only pseudo labels.

![](histograms.jpg)

One notable thing is the bias to ca 5 deg. pitch. This was important for maximizing the benchmark score on AFLW2k-3D and matching my networks performances with training on FaceSynthetics.

Beware that the default settings for ReplicantFace will not produce this distribution. The current data stems from a combination of the default with a correction with more samples which I added later to achieve the pitch bias. See below.

The size distribution of the bounding boxes is shown next. The faces only fill half of the images to allow for
shift augmentation with visible background.

![](boundingboxsizes.jpg)

Considering the **depth coordinates** of landmarks, we have to recognize that their relative
differences across the face is essential. On the other hand, their center of mass will in general be different
from dataset to dataset or even face to face. Since there is next to no information in the images to derive an absolute
depth from, an ML algorithm could learn at most an offset for the entire dataset.
Therefore, caution must be exercised especially when combining datasets of different origin.

This dataset comes with the point between the eyes centered at zero. I.e. by
```Python
def depth_centered_keypoints(kpts):
    eye_corner_indices = [45, 42, 39, 36]
    center = np.average(kpts[:,eye_corner_indices], axis=1)
    kpts = np.array(kpts, copy=True)
    kpts[2] -= center[2]
    return kpts
```
When labels are made consistent by such means, there should be no problems. Of course an appropriate loss can also be
used.


## Details about the pitch correction

75k samples were created with the default setting at the time of this writing.
Then 25k more were created with the following patch to `randomize_pose.py`:
```diff
@@ -130,9 +130,9 @@ def randomize_pose(hum : Human):
         print ("selected pose: ", new_pose)
 
     # Change head direction
-    heading = 70./180.*pi*random_beta_11(3.)
-    pitch = 45./180.*pi*random_beta_11(3.)
-    roll = 30./180.*pi*random_beta_11(3.)
+    heading = 70./180.*pi*random_beta_11(4.)
+    pitch = (random_beta_11(4.)*20.+3.) /180.*pi # Positive pitch = looking down
+    roll = 10./180.*pi*random_beta_11(4.)
     bones = rig.pose.bones
     headbone = bones['head']
     neckbone = bones['neck']
@@ -148,22 +148,21 @@ def randomize_pose(hum : Human):
 
 
 def randomize_camera(cam : bpy.types.Object, hum_object : bpy.types.Object, env_cam : bpy.types.Object):
-    # Camera parameters
+    assert cam.parent is not None
+
     update_child_of_constraint(cam.parent, hum_object, 'head')
 
     rig_heading = hum_object.rotation_euler[2]
 
-    if False: #random.randint(0,100) == 0:
-        heading = random.uniform(-pi,pi)
-    else:
-        heading = random_beta_11(3.)*100.*pi/180.
-    pitch = random_beta_11(3.)*30.*pi/180.
-    if 1:
-        cam.parent.rotation_euler[2] = rig_heading + heading
-        cam.parent.rotation_euler[1] = 0.
-        cam.parent.rotation_euler[0] = pi/2. + pitch
+    heading = random_beta_11(4.)*70.*pi/180.
+    pitch = (random_beta_11(4.)*5. - 2.)*pi/180. # positive values make the camera look up and thus cause positive pitch of the face.
+    cam.parent.rotation_euler[2] = rig_heading + heading
+    cam.parent.rotation_euler[1] = 0.
+    cam.parent.rotation_euler[0] = pi/2. + pitch # This is *not* added to the face pitch
+
     distance = cam.location[2]
     cam.data.dof.focus_distance = distance
+    
     # Env cam
     env_cam.data.lens = random.choice([25.,30.,50.,70.])
```