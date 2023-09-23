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

pyutilpath = joinpath(@__DIR__, "Utils")
pushfirst!(PyVector(pyimport("sys")."path"), pyutilpath )

const np = PyNULL()
const cv = PyNULL()
# const SscPy = PyNULL()
SscPy = pyimport("PySSCFeatures")

function __init__()
    copy!(np, pyimport("numpy"))
    copy!(cv, pyimport("cv2"))
    # copy!(SscPy, pyimport("PySSCFeatures"))
end


ssc = SscPy."ssc"


include("services/OpenCVFeatures.jl")

end