"""
    undistortImage(image, K, D)

OpenCV undistort image for equi and radtan distortion models
https://github.com/ethz-asl/kalibr/wiki/supported-models
K - Input camera matrix K=[fx 0 cx; 0 fy cy; 0 0 1]
D (equi) - Input vector of distortion coefficients (k1,k2,k3,k4)
D (radtan) - Input vector of distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements. 
    If the vector is NULL/empty, the zero distortion coefficients are assumed.
"""
function undistortImage(img, K, D, P=K; distortion_model=:equi, new_wh = size(img'))

    # cimg = RGB{N0f8}.(img)
    # pyimg = reinterpret.(UInt8, PermutedDimsArray(channelview(cimg), (2,3,1)))
    pyimg = collect(reinterpret(UInt8, img))
    
    if distortion_model == :equi
        undist_pyimg = cv.fisheye.undistortImage(pyimg, K, D, Knew=P, new_size=new_wh)

    elseif distortion_model == :radtan
        undist_pyimg = cv.undistort(pyimg, K, distCoeffs=D, newCameraMatrix=P)

    else
        @error("only 'equi' and 'radtan' supported")
        return img
    end
    # undist_img = reinterpretc(RGB{N0f8}, PermutedDimsArray(undist_pyimg, (3,1,2)))
    undist_img = Gray.(reinterpret(N0f8, undist_pyimg))

    return undist_img
end

function undistortImage(img, K, D, R, P; distortion_model=:equi, new_wh=size(img'))

    # cimg = RGB{N0f8}.(img)
    # pyimg = reinterpret.(UInt8, PermutedDimsArray(channelview(cimg), (2,3,1)))

    pyimg = collect(reinterpret(UInt8, img))
    
    if distortion_model == :equi
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, R, P, new_wh, cv.CV_32FC1)#cv.CV_16SC2)
        undist_pyimg = cv.remap(
            collect(reinterpret(UInt8, img)),
            map1,
            map2,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT
        )

    elseif distortion_model == :radtan
        mapx, mapy = cv.initUndistortRectifyMap(K, D, R, P, new_wh, 5)
        undist_pyimg = cv.remap(pyimg, mapx, mapy, cv.INTER_LINEAR)

    else
        @error("only 'equi' and 'radtan' supported")
        return img
    end
    # undist_img = reinterpretc(RGB{N0f8}, PermutedDimsArray(undist_pyimg, (3,1,2)))
    undist_img = Gray.(reinterpret(N0f8, undist_pyimg))

    return undist_img
end

function undistortImage(
    fg::AbstractDFG,
    entry::BlobEntry,
    K,
    D,
    R,
    P; 
    new_wh=nothing,
    distortion_model=:equi
)

    blob = getBlob(fg, entry)
    img = unpackBlob(DFG.format"PNG", blob)
    isnothing(new_wh) && (new_wh = size(img'))

    undistortImage(img, K, D, R, P; new_wh, distortion_model)
end

function undistortImage(
    fg::AbstractDFG,
    entry::BlobEntry,
    K,
    D;
    P=K,
    new_wh=nothing,
    distortion_model=:equi
)
    blob = getBlob(fg, entry)
    img = unpackBlob(DFG.format"PNG", blob)
    isnothing(new_wh) && (new_wh = size(img'))

    return undistortImage(img, K, D; distortion_model, P, new_wh)
end
