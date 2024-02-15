
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


"""
    $SIGNATURES

Recovers the relative camera rotation and the translation from 
corresponding points in two images, using an estimated 
fundamental matrix (i.e. epipolar) and chirality check.

Notes
- Interally employs cv's RANSAC.

See also: [`getPoseEssential`](@ref)
"""
function getPoseFundamental(p1, p2, K)
  @error "WIP Result may be broken."
  # FIXME change to `cv.findFundamentalMat`: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga59b0d57f46f8677fb5904294a23d404a
  # Calculates a fundamental matrix from the corresponding points in two images.
  F, fmat_mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 2.0, 0.999, nothing)  
  # get essential matrix from the fundamental matrix using the camera intrinsics
  # https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
  E = K'*F*K

  # _p1 = p1[findall(essmat_mask[:] .== 0x01), :]
  # _p2 = p2[findall(essmat_mask[:] .== 0x01), :]
  # rv, R, t, recpose_mask = cv.recoverPose(E, _p1, _p2, K, nothing)
  rv, R, t, recpose_mask = cv.recoverPose(E, p1, p2, K, nothing)
  return rv, R, t, recpose_mask
end

"""
    $SIGNATURES

Recovers the relative camera rotation and the translation 
from corresponding points in two images, using the essential 
matrix and chirality check.

Notes
- Interally employs cv's RANSAC.

See also: [`getPoseFundamental`](@ref)
"""
function getPoseEssential(p1, p2, K)
  # @warn "consider using getPoseFundamental rather than current getPoseEssential" maxlog=5
  # Calculates an essential matrix from the corresponding points in two images.
  E, essmat_mask = cv.findEssentialMat(p1, p2, K, cv.RANSAC, 0.999, 1.0, nothing)
  # Returns the number of inliers that pass the check.
  
  # _p1 = p1[findall(essmat_mask[:] .== 0x01), :]
  # _p2 = p2[findall(essmat_mask[:] .== 0x01), :]
  # rv, R, t, recpose_mask = cv.recoverPose(E, _p1, _p2, K, nothing)
  rv, R, t, recpose_mask = cv.recoverPose(E, p1, p2, K, nothing)
  return rv, R, t, recpose_mask
end

"""
    $SIGNATURES

Wrapper to Python OpenCV [`goodFeaturesToTrack`](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541).

On `feature_params`:
- `maxCorners`: 	Maximum number of corners to return. If there are more corners than are found, the strongest of them is returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
- `qualityLevel`:	 Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
- `minDistance`: 	Minimum possible Euclidean distance between the returned corners.
- `mask`:	Optional region of interest. If the image is not empty (it needs to have the type CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.
- `blockSize`:	Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. See cornerEigenValsAndVecs .
- `useHarrisDetector`:	Parameter indicating whether to use a Harris detector (see cornerHarris) or cornerMinEigenVal.
- `k`:	Free parameter of the Harris detector.

Notes
- [Shi-Tomasi Corner Detector and Good Features to Track, OpenCV Tutorial](https://docs.opencv.org/5.x/d4/d8c/tutorial_py_shi_tomasi.html)
"""
function goodFeaturesToTrack(
  im1, 
  feature_params; 
  mask=nothing
)
  _mask = isnothing(mask) ? nothing : collect(reinterpret(UInt8, mask))
  cv.goodFeaturesToTrack(collect(reinterpret(UInt8, im1)); mask=_mask, feature_params...)
end

"""
    $SIGNATURES

Notes
- [ORB CV Tutorial](https://docs.opencv.org/5.x/d1/d89/tutorial_py_orb.html)
"""
function goodFeaturesToTrackORB(
  im1; 
  mask=nothing, 
  orb = cv.ORB_create(), 
  downsample::Int=1, 
  tolerance::Real = 0.1
) 
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
    pyssc(kp, orb.getMaxFeatures() ÷ downsample, tolerance, cols, rows)
  else
    kp
  end

  # compute the descriptors with ORB
  kp_, des = orb.compute(img, sel_kp)

  return kp_, des
end

function getPoseSIFT(imgA, imgB, K; mask=nothing)
    # Initiate SIFT detector
  sift = cv.SIFT_create()
  # sift = cv.ORB_create()
  # find the keypoints and descriptors with SIFT
  img1 = collect(reinterpret(UInt8, imgA))
  img2 = collect(reinterpret(UInt8, imgB))
  kp1, des1 = sift.detectAndCompute(img1, mask)
  kp2, des2 = sift.detectAndCompute(img2, mask)
  # BFMatcher with default params
  bf = cv.BFMatcher()
  matches = bf.knnMatch(des1,des2,k=2)
  # Apply ratio test
  good = []
  for mat in matches
      m = mat[1]  
      n = mat[2]  
      if m.distance < 0.75 * n.distance
          push!(good, (distance=m.distance, queryIdx=m.queryIdx+1, trainIdx=m.trainIdx+1))
      end
  end

  pairs = map(good) do m
      kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
  end

  p1 = mapreduce(vcat, first.(pairs)) do p
    [p[1] p[2];]
  end

  p2 = mapreduce(vcat, last.(pairs)) do p
    [p[1] p[2];]
  end

  E, essmat_mask = cv.findEssentialMat(p2, p1, K, cv.RANSAC, 0.999, 1.0, nothing)
  rv, R, t, recpose_mask = cv.recoverPose(E, p2, p1, K, nothing)

  return (rv=rv, R=R, t=t, recpose_mask=recpose_mask, p1=p1, p2=p2, good=good)
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
    
    rv, R, t, recpose_mask = getPose(p2, p1, K)
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

"""
    $SIGNATURES

Track features across neighboring frames +-1, +-2, ...

Notes
- expecting odd number of images for forward backward tracking from center image.
"""
function trackFeaturesForwardsBackwards(
  imgs, 
  feature_params, 
  lk_params; 
  mask=nothing, 
  orb = cv.ORB_create()
)
  len = length(imgs)
  @assert isodd(len) "expecting odd number of images for forward backward tracking from center image."
  
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




## To consolidate with above functions

"""
ORB_create([, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]) -> retval
.   @brief The ORB constructor
.   
@param nfeatures The maximum number of features to retain.

@param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical
.       pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor
.       will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor
.       will mean that to cover certain scale range you will need more pyramid levels and so the speed
.       will suffer.

@param nlevels The number of pyramid levels. The smallest level will have linear size equal to
.       input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).

@param edgeThreshold This is size of the border where the features are not detected. It should
.       roughly match the patchSize parameter.

@param firstLevel The level of pyramid to put source image to. Previous layers are filled
.       with upscaled source image.

@param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The
.       default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
.       so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
.       random points (of course, those point coordinates are random, but they are generated from the
.       pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
.       rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
.       output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
.       denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
.       bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).

@param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features
.       (the score is written to KeyPoint::score and is used to retain best nfeatures features);
.       FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
.       but it is a little faster to compute.

@param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller
.       pyramid layers the perceived image area covered by a feature will be larger.

@param fastThreshold the fast threshold

detect(image[, mask]) -> keypoints
.   @brief Detects keypoints in an image (first variant) or image set (second variant).
.       @param image Image.
.       @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set
.       of keypoints detected in images[i] .
.       @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
.       matrix with non-zero values in the region of interest.

compute(image, keypoints[, descriptors]) -> keypoints, descriptors
.   @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
.       (second variant).
.   
.       @param image Image.
.       @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be
.       computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint
.       with several dominant orientations (for each orientation).
.       @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are
.       descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the
.       descriptor for keypoint j-th keypoint.

"""
function cv_create_orb_descriptor(
    img;
    nfeatures = 500,
    nlevels = 8,
    fastThreshold = 20,
    edgeThreshold = 31,
    firstLevel = 0,
    patchSize = 31,
    WTA_K = 2,
    scaleFactor = 1.2,
    scoreType = 0,
    mask = nothing,
)       

    pyimg = collect(reinterpret(UInt8, img))
    orb = py"cv.ORB_create"(
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

    kp = orb.detect(pyimg, mask)
    kp_cv, des_cv = orb.compute(pyimg, kp)

    keypoints = map(
        kp->(
            angle=kp.angle,
            pt=kp.pt,
            response=kp.response,
            octave=kp.octave,
            size=kp.size
        ), 
        [kp_cv...]
    )

    des_bv = map(eachrow(des_cv)) do d
        bv = BitVector()
        bv.chunks = reinterpret(UInt64, d)
        bv.len = 256
        return bv
    end

    return kp_cv, keypoints, des_cv, des_bv
end

function cv_create_orb_descriptor(
    img,
    keypoints;
    nfeatures = 500,
    nlevels = 8,
    fastThreshold = 20,
    edgeThreshold = 31,
    firstLevel = 0,
    patchSize = 31,
    WTA_K = 2,
    scaleFactor = 1.2,
    scoreType = 0,
)       

    pyimg = collect(reinterpret(UInt8, img))
    orb = py"cv.ORB_create"(
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

    kp_cv, des_cv = orb.compute(pyimg, keypoints)

    keypoints = map(kp_cv->(angle=kp_cv.angle,pt=kp_cv.pt), [kp_cv...])

    des_bv = map(eachrow(des_cv)) do d
        bv = BitVector()
        bv.chunks = reinterpret(UInt64, d)
        bv.len = 256
        return bv
    end

    return kp_cv, keypoints, des_cv, des_bv
end

# create BFMatcher object
"""
Brute-force descriptor matcher.

For each descriptor in the first set, this matcher finds the closest descriptor in the second set by trying each one. This descriptor matcher supports masking permissible matches of descriptor sets.

Usage in Julia:
```julia
bfm = cv_create_BFMatcher()
matches = bfm.match(query_desc,train_desc)
smatches = matches[sortperm((s->s.distance).(matches))]

# best match
smatches[1].queryIdx, smatches[1].train_desc
```

`matches = bfmatcher.match(query, train, ...)` finds the best match for each descriptor from a query set.

    Parameters
        queryDescriptors	Query set of descriptors.
        trainDescriptors	Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        matches	Matches. If a query descriptor is masked out in mask , no match is added for this descriptor. So, matches size may be smaller than the query descriptors count.
        mask	Mask specifying permissible matches between an input query and train matrices of descriptors.
    
    In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at<uchar>(i,j) is non-zero.
    
"""
cv_create_BFMatcher(;crossCheck=true) = py"cv.BFMatcher"(py"cv.NORM_HAMMING", crossCheck=crossCheck)


function getOrbEntry(fg, label, cv_masks)
    entry = getBlobEntry(fg, label, r"^cam")
    blob = getBlob(fg, entry)
    img = unpackBlob(DFG.format"PNG", blob)

    mask = cv_masks[maskLabel(entry)]
    kp_cv, keypoints, des_cv, des_bv = cv_create_orb_descriptor(
        img;
        nfeatures = 200,
        nlevels = 1,
        scaleFactor = 1.1,
        scoreType = 1,
        mask
    )
    smalldata = Dict{Symbol, DFG.SmallDataTypes}()
    push!(smalldata, :keypoints=>JSON3.write(keypoints))
    push!(smalldata, :descriptors=>base64encode.(eachrow(des_cv)))
    metadata = base64encode(JSON3.write(smalldata))
    orbentry = BlobEntry(entry; label=:features, metadata, description = entry.description*" features")
    label=>orbentry
end

function feature(entry::BlobEntry)

    meta = JSON3.read(String(base64decode(entry.metadata)))
    des_cv = base64decode.(meta.descriptors)
    
    des_bv = map(des_cv) do d
        bv = BitVector()
        bv.chunks = reinterpret(UInt64, d)
        bv.len = 256
        return bv
    end

    keypoints = JSON3.read(
        meta.keypoints,
        Vector{NamedTuple{
            (:angle, :pt, :response, :octave, :size),
            Tuple{Float64, Tuple{Float64, Float64}, Float64, Int64, Float64}
        }}
    )

    return (keypoints=keypoints, des_bv=des_bv)
end

function matchFeatures(feat1, feat2, threshold::Float64 = 0.1, onlyLevel = 0) 
    #@param onlyLevel level in middle of range matches smaller and larger
    keypoints_1 = getproperty.(feat1.keypoints,:pt)
    if !isnothing(onlyLevel)
        keypoints_1 = map(feat1.keypoints) do kp
            kp.octave == onlyLevel ? kp.pt : missing
        end        
        misidx = filter(>(0), (!).(ismissing.(keypoints_1)) .* eachindex(keypoints_1))
        filter!(!ismissing, keypoints_1)
    end

    desc_1 = feat1.des_bv
    if !isnothing(onlyLevel)
        desc_1 = map(feat1.des_bv, feat1.keypoints) do des,kp
            kp.octave == onlyLevel ? des : missing
        end
        filter!(!ismissing, desc_1)
    end
    
    keypoints_2 = getproperty.(feat2.keypoints,:pt)
    desc_2 = feat2.des_bv

    smaller = desc_1
    larger = desc_2
    s_key = keypoints_1
    l_key = keypoints_2
    order = false
    if length(desc_1) > length(desc_2)
        smaller = desc_2
        larger = desc_1
        s_key = keypoints_2
        l_key = keypoints_1
        order = true
    end
    hamming_distances = [ImageFeatures.hamming_distance(s, l) for s in smaller, l in larger]
    matches = typeof(keypoints_1)[]
    matches_idx = Tuple{Int,Int}[]
    scores = Float64[]
    for i in 1:length(smaller)
        if any(hamming_distances[i, :] .< threshold)
            id_min = argmin(hamming_distances[i, :])
            push!(matches, order ? [l_key[id_min], s_key[i]] : [s_key[i], l_key[id_min]])
            if !isnothing(onlyLevel)
                #FIXME test if this is correct
                push!(matches_idx, order ? (misidx[id_min], i) : (misidx[i], id_min))
                
            else
                push!(matches_idx, order ? (id_min, i) : (i, id_min))
            end
            push!(scores, hamming_distances[i, id_min])
            hamming_distances[:, id_min] .= 1.0
        end
    end
    matches, matches_idx, scores
end

##
