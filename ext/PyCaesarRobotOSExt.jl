module PyCaesarRobotOSExt

using PyCall
using OpenSSL # workaround attempt, https://github.com/JuliaWeb/OpenSSL.jl/issues/9
using RobotOS
using Caesar
using Dates
using DocStringExtensions



## FIXME DEPS
using Colors
using ImageMagick, FileIO
using Images
# likely packages needed to ingest various different sensor data types and serialize them for NVA
using TimeZones
using Random
using JSON
using BSON, Serialization
using FixedPointNumbers
using StaticArrays





import Base: convert

# FIXME DEPRECATE UPGRADE TBD
import Unmarshal: unmarshal

import Caesar._PCL as _PCL

# weakdeps type and memver prototype import for overwritten definition pattern 
import PyCaesar: RosbagSubscriber, RosbagWriter
import PyCaesar: loop!, getROSPyMsgTimestamp, nanosecond2datetime



##




@rosimport std_msgs.msg: Header

# standard types
@rosimport sensor_msgs.msg: PointCloud2
@rosimport sensor_msgs.msg: LaserScan
@rosimport sensor_msgs.msg: CompressedImage
@rosimport sensor_msgs.msg: Image
@rosimport sensor_msgs.msg: Imu
@rosimport tf2_msgs.msg: TFMessage
@rosimport geometry_msgs.msg: TransformStamped
@rosimport geometry_msgs.msg: Transform
@rosimport geometry_msgs.msg: Vector3
@rosimport geometry_msgs.msg: Quaternion
@rosimport nav_msgs.msg: Odometry

# FIXME, note this is not working, functions using Main. as workaround
@rosimport std_msgs.msg: Header
@rosimport sensor_msgs.msg: PointField
@rosimport sensor_msgs.msg: PointCloud2

# FIXME not sure if this should be run here

# thismodule(m::Module, args...) = (@info("this module is getting", m); m)
# macro thismodule(args...)
#   :( thismodule(__module__) )
# end
rostypegen(@__MODULE__)


include("Utils/RosbagSubscriber.jl")
include("services/ROSConversions.jl")
include("services/PCLROSConversions.jl")
include("services/IngestROSSetup.jl")

end