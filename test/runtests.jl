
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

# using RobotOS