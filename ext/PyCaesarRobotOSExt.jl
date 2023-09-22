module PyCaesarRobotOSExt

using PyCall
using OpenSSL # workaround attempt, https://github.com/JuliaWeb/OpenSSL.jl/issues/9
using RobotOS
using Dates
using DocStringExtensions

using Colors
using ImageMagick, FileIO
# using Images
# likely packages needed to ingest various different sensor data types and serialize them for NVA
using TimeZones
using Random
using JSON3
using BSON
# , Serialization
using FixedPointNumbers
using StaticArrays

# dont load Caesar until after rostypegen
using Caesar
# import Caesar: unmarshal

# FIXME DEPRECATE UPGRADE TBD
import Unmarshal: unmarshal

import Caesar._PCL as _PCL

import Base: convert

# weakdeps type and memver prototype import for overwritten definition pattern 
import PyCaesar: RosbagSubscriber, RosbagWriter
import PyCaesar: loop!, getROSPyMsgTimestamp, nanosecond2datetime
import PyCaesar: handleMsg!, handleMsg_OVRLPRECOMP!

##

@rosimport std_msgs.msg: Header

@rosimport tf2_msgs.msg: TFMessage

@rosimport nav_msgs.msg: Odometry

# standard types
@rosimport sensor_msgs.msg: PointCloud2
@rosimport sensor_msgs.msg: PointField
@rosimport sensor_msgs.msg: LaserScan
@rosimport sensor_msgs.msg: CompressedImage
@rosimport sensor_msgs.msg: Image
@rosimport sensor_msgs.msg: Imu
@rosimport sensor_msgs.msg: NavSatFix
@rosimport sensor_msgs.msg: RegionOfInterest
@rosimport sensor_msgs.msg: CameraInfo

@rosimport geometry_msgs.msg: Vector3
@rosimport geometry_msgs.msg: Quaternion
@rosimport geometry_msgs.msg: Point
@rosimport geometry_msgs.msg: Pose
@rosimport geometry_msgs.msg: Transform
@rosimport geometry_msgs.msg: TransformStamped
@rosimport geometry_msgs.msg: Twist
@rosimport geometry_msgs.msg: TwistWithCovariance
@rosimport geometry_msgs.msg: TwistWithCovarianceStamped

# load type definitions into this modules context
rostypegen(@__MODULE__)

##

include("Utils/RosbagSubscriber.jl")
include("services/ROSConversions.jl")
include("services/PCLROSConversions.jl")
include("services/IngestROSSetup.jl")

end