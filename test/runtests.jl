
using Test
using PyCaesar

##

@test PyCaesar.version() isa VersionNumber

##

using Images
using TestImages

##

img = testimage("cameraman") # without extension works

kps, dsc = PyCaesar.goodFeaturesToTrackORB(img)

@test 1 < length(kps)
@test dsc isa Matrix

##

# having trouble with Github Action ROS install
# using RobotOS
import Caesar._PCL as _PCL

wPC  = _PCL.PointCloud()
wPC2 = _PCL.PCLPointCloud2(wPC)
# rmsg = PyCaesar.toROSPointCloud2(wPC2);


##