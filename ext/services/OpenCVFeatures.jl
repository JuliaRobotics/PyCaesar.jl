


# # lk_params = ( winSize  = (19, 19), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# lk_params = ( winSize  = (19, 19), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
# feature_params = (maxCorners = 1000, qualityLevel = 0.01, minDistance = 8, blockSize = 19 )

function calcFlow(jl_img0, jl_img1, p0, lk_params, back_threshold = 1.0)
    # https://www.programcreek.com/python/example/89363/cv2.calcOpticalFlowPyrLK
    img0 = collect(reinterpret(UInt8, jl_img0))
    img1 = collect(reinterpret(UInt8, jl_img1))
    status = zeros(UInt8, length(p0))
    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, nothing; lk_params...)
    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, nothing; lk_params...)
    d = maximum(reshape(abs.(p0 .- p0r), :, 2); dims=2)#.reshape(-1, 2).max(-1)
    status = d .< back_threshold
    return p1, status
end


function getPose(p1, p2, K)
    # Calculates an essential matrix from the corresponding points in two images.
    E, essmat_mask = cv.findEssentialMat(p1, p2, K, cv.RANSAC, 0.999, 1.0, nothing)
    # Recovers the relative camera rotation and the translation from an estimated 
    # essential matrix and the corresponding points in two images, using chirality check.
    # Returns the number of inliers that pass the check.
    rv, R, t, recpose_mask = cv.recoverPose(E, p1, p2, K, nothing)
    return rv, R, t, recpose_mask
end

function goodFeaturesToTrack(im1, feature_params; mask=nothing) 
  cv.goodFeaturesToTrack(collect(reinterpret(UInt8, im1)), mask=collect(reinterpret(UInt8,mask)), feature_params...)
end

function goodFeaturesToTrackORB(im1; mask=nothing, orb = cv.ORB_create(), downsample::Int=1, tolerance::Real = 0.1) 
  # gray = cv2.cvtColor(im1,cv.COLOR_BGR2GRAY)
  # kypts, decrs = orb.detectAndCompute(gray,None)
  # https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
  # find the keypoints with ORB
  img = collect(reinterpret(UInt8, im1))
  kp = if isnothing(mask)
    orb.detect(img)
  else
    orb.detect(img, collect(reinterpret(UInt8,mask)))
  end

  sel_kp = if 1 < downsample
    # downselect a better distribution of features
    rows, cols = size(img,1), size(img,2)
    ssc(kp, orb.getMaxFeatures() ÷ downsample, tolerance, cols, rows)
  else
    kp
  end

  # compute the descriptors with ORB
  kp_, des = orb.compute(img, sel_kp)

  return kp_, des
end

function combinePlot(ref_img, overlay_img)
    combine = map(zip(reinterpret(Gray{N0f8}, ref_img), reinterpret(Gray{N0f8}, overlay_img))) do (a,b)
        RGB(a, b, a)
    end
    f = image(rotr90(combine), axis = (aspect = DataAspect(),))
    return f
end

function getPose(im1, im2, K, feature_params, lk_params; mask=nothing)

    p1 = goodFeaturesToTrack(im1, feature_params; mask)
    
    p2, flow_status = calcFlow(im1, im2, p1, lk_params)
    
    # only keep good points
    p1 = p1[flow_status, :, :]
    p2 = p2[flow_status, :, :]
    
    rv, R, t, recpose_mask = getPose(p1, p2, K)
    return rv, R, t, recpose_mask
end


function trackFeaturesFrames(
  feats0, 
  jl_imgs0_n::AbstractVector{<:AbstractMatrix},
  lk_params;
  mask = nothing
)
  # https://www.programcreek.com/python/example/89363/cv2.calcOpticalFlowPyrLK
  imgs = reinterpret.(UInt8, jl_imgs0_n) .|> collect

  tracks = []
  # status = zeros(UInt8, length(feats0))
  for (i,img) in enumerate(imgs)
    # skip first image which is assumed to coincide with feats0 (good features on first frame)
    i == 1 ? continue : nothing
    p1, _st, _err = cv.calcOpticalFlowPyrLK(imgs[1], img, feats0, nothing; lk_params...) # collect(reinterpret(UInt8,mask))
    push!(tracks, p1)
  end

  return tracks
end


function trackFeaturesForwardsBackwards(imgs, feature_params, lk_params; mask=nothing, orb = cv.ORB_create())

  len = length(imgs)
  @assert isodd(len) "expecting odd number of images for forward backward tracking from center image"
  
  cen = floor(Int, len/2) + 1

  img_tracks = Dict{Int,Vector{Vector{Float64}}}()
  dscs0 = Dict{Int,Tuple{Float64, Vector{Int}}}()

  # use orb /w descriptors || good features
  feats0 = if true
    kpts, dscs = goodFeaturesToTrackORB(imgs[cen]; mask, orb)
    img_tracks[0] = [[kpts[k].pt[1];kpts[k].pt[2]] for k in 1:length(kpts)]
    # legacy fill feats0 for tracking
    feats0_ = zeros(Float32,length(kpts),1,2)
    for k in 1:length(kpts)
      feats0_[k,1,1] = kpts[k].pt[1]
      feats0_[k,1,2] = kpts[k].pt[2]
      dscs0[k] = (kpts[k].angle, Int.(dscs[k,:][:]))
    end
    feats0_
  else
    feats0_ = goodFeaturesToTrack(imgs[cen], feature_params; mask)
    img_tracks[0] = [feats0_[k,:,:][:] for k in 1:size(feats0_,1 )]
    feats0_
  end

  tracks = trackFeaturesFrames(feats0, imgs[cen:end], lk_params; mask)
  for (i,tr) in enumerate(tracks)
    isnothing(tr) && continue
    img_tracks[i] = [tr[k,:,:][:] for k in 1:size(tr,1)]
  end

  tracks = trackFeaturesFrames(feats0, imgs[cen:-1:1], lk_params; mask)
  for (i,tr) in enumerate(tracks)
    img_tracks[-i] = [tr[k,:,:][:] for k in 1:size(tr,1)]
  end

  return img_tracks, dscs0
end


function makeBlobFeatureTracksPerImage_FwdBck!(
  dfg::AbstractDFG,
  vlbs_fwdbck::AbstractVector{Symbol},
  imgBlobKey,
  blobstorelbl::Symbol,
  blobLabel::Symbol = Symbol("IMG_FEATURE_TRACKS_FWDBCK_$(length(vlbs_fwdbck))_KLT"),
  descLabel::Symbol = Symbol("IMG_FEATURE_ANG_ORB");
  feature_params,
  lk_params,
  mask,
  orb = cv.ORB_create()
)
  # good features to track
  kfs = (s->getData(dfg, s, imgBlobKey)).(vlbs_fwdbck)
  imgs = kfs .|> (eb)->unpackBlob(MIME(eb[1].mimeType), eb[2])
  # track features across neighboring frames +-1, +-2, ...
  img_tracks, dscs0 = trackFeaturesForwardsBackwards(imgs, feature_params, lk_params; mask, orb)
  
  center_vlb = vlbs_fwdbck[1+floor(Int,length(vlbs_fwdbck)/2)]

  # store feature keypoints and tracks
  Caesar.addDataImgTracksFwdBck!(
    dfg,
    center_vlb,
    blobstorelbl,
    blobLabel,
    "",
    img_tracks,
  )

  # only store descriptors if available
  if 0 < length(dscs0)
    # store feature angles and descriptors in a separate blob
    blob = Vector{UInt8}(JSON3.write(dscs0))
    entry = BlobEntry(;
      label = descLabel,
      blobstore = blobstorelbl,
      hash = bytes2hex(sha256(blob)),
      origin = "",
      size = length(blob),
      description = "Image feature angles and ORB descriptors from cv2.py",
      mimeType = "application/octet-stream/json; _type=JuliaLang.$(typeof(dscs0))"
    )
    addData!(dfg, center_vlb, entry, blob)
  end
end

function makeORBParams(
    feature_params;
    nfeatures = feature_params.maxCorners,
    nlevels = 1,
    fastThreshold = 20,
    edgeThreshold = 31,
    firstLevel = 0,
    patchSize = 31,
    WTA_K = 2,
    scaleFactor = 1.1,
    scoreType = 1,
)

  orb = cv.ORB_create(
    nlevels = nlevels,
    fastThreshold = fastThreshold,
    edgeThreshold = edgeThreshold,
    firstLevel = firstLevel,
    patchSize = patchSize,
    WTA_K = WTA_K,
    nfeatures = nfeatures,
    scaleFactor = scaleFactor,
    scoreType = scoreType,
  )
  # orb.setMaxFeatures(feature_params.maxCorners)
  # orb.setPatchSize(feature_params.blockSize)
  # orb.setNLevels(1)
  # orb.setScoreType(1) # FAST
  return orb
end




#