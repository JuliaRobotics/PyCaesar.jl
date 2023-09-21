module PyCaesarImagesExt

using PyCall
using Images
# using PyCaesars
# using ImageFeatures
using DistributedFactorGraphs
using ProgressMeter
using JSON3
using TensorCast
using SHA: sha256

import PyCaesar: calcFlow, getPose, goodFeaturesToTrack, goodFeaturesToTrackORB, combinePlot
import PyCaesar: trackFeaturesFrames, trackFeaturesForwardsBackwards
import PyCaesar: makeBlobFeatureTracksPerImage_FwdBck!, makeORBParams

export calcFlow, getPose, goodFeaturesToTrack, goodFeaturesToTrackORB, combinePlot
export trackFeaturesFrames, trackFeaturesForwardsBackwards, makeBlobFeatureTracksPerImage_FwdBck!, makeORBParams

const np = PyNULL()
const cv = PyNULL()

function __init__()
    copy!(np, pyimport("numpy"))
    copy!(cv, pyimport("cv2"))
end

pyutilpath = joinpath(@__DIR__, "Utils")
pushfirst!(PyVector(pyimport("sys")."path"), pyutilpath )

SscPy = pyimport("PySSCFeatures")
ssc = SscPy."ssc"


include("services/OpenCVFeatures.jl")

end