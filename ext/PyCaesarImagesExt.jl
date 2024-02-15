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
using Caesar
using DocStringExtensions

import PyCaesar: calcFlow, goodFeaturesToTrack, goodFeaturesToTrackORB, combinePlot
import PyCaesar: trackFeaturesFrames, trackFeaturesForwardsBackwards
import PyCaesar: makeBlobFeatureTracksPerImage_FwdBck!, makeORBParams
import PyCaesar: pycv
import PyCaesar: getPoseEssential, getPoseFundamental
import PyCaesar: getPose # deprecating
import PyCaesar: getPoseSIFT # deprecating

export calcFlow, getPoseEssential, goodFeaturesToTrack, goodFeaturesToTrackORB, combinePlot
export trackFeaturesFrames, trackFeaturesForwardsBackwards, makeBlobFeatureTracksPerImage_FwdBck!, makeORBParams

pyutilpath = joinpath(@__DIR__, "Utils")
pushfirst!(PyVector(pyimport("sys")."path"), pyutilpath )

const np = PyNULL()
const cv = PyNULL()
# const SscPy = PyNULL()
SscPy = pyimport("PySSCFeatures")

# reset the pointers between precompile and using
function __init__()
    copy!(np, pyimport("numpy"))
    copy!(cv, pyimport("cv2"))
    # copy!(SscPy, pyimport("PySSCFeatures"))
end

"""
    $SIGNATURES

Function to expose the modules internal cv pointer.
"""
pycv() = cv

pyssc = SscPy."ssc"


include("services/OpenCVFeatures.jl")


# deprecation
@deprecate getPose(p1, p2, K) getPoseEssential(p1, p2, K)


end