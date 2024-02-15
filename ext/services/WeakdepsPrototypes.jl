
# prototypes for functions in this PyCaesar extensions

# Images.jl
function calcFlow end
function getPose end
function getPoseSIFT end
function getPoseEssential end
function getPoseFundamental end
function goodFeaturesToTrack end
function goodFeaturesToTrackORB end
function combinePlot end
function trackFeaturesFrames end
function trackFeaturesForwardsBackwards end
function makeBlobFeatureTracksPerImage_FwdBck! end
function makeORBParams end
function pycv end

# RobotOS.jl
function handleMsg! end
function handleMsg_OVRLPRECOMP! end
function loop! end
function getROSPyMsgTimestamp end
function nanosecond2datetime end
function toROSImage end