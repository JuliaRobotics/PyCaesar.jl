
"""
    $SIGNATURES

Standardized format callback for ros odometry messages.
"""
function handleMsg!(
  msg::nav_msgs.msg.Odometry, 
  dfg, 
  state::SystemState, 
  options=Dict()
)
  @info "handleOdometry!" maxlog=10

  # handle case for first message
  if !haskey(state.workspace, "Odometry")
    state.workspace["Odometry"] = state.prv_variable
  end
  
  msgd = unmarshal(msg)
  
  # quick and dirty synchronization to keyframe message.
  # NOTE storing only the first msg after a new pose was created
  # @info "WHY" state.workspace["Odometry"] state.prv_variable
  if state.workspace["Odometry"] == state.prv_variable
    return nothing
  end

  # timestamp = Float64(msg.header.stamp.secs) + Float64(msg.header.stamp.nsecs)/10^9
  # @info "[$timestamp] TFMessage sample on $(state.cur_variable)"

  # unmarshal msg
  # imgd = Caesar.unmarshal(msg)
  # img = Caesar.toImage(msg)

  blob_lbl = ODOMSG_BLOBNAME()
  blob = JSON.json(msgd) |> Vector{UInt8}
  mime = "application/json/"*msgd["_type"]

  # @info "TF2Message" msg.transforms[1].header.seq msg.transforms[1].transform.translation
  @show state.pushBlobs
  if state.pushBlobs
    blobid = addData(dfg, blob_lbl, blob)
    
    # not super well synched, but good enough for basic demonstration
    addBlobEntry!(
      dfg,
      # next of pair for now as rest works like that
      state.cur_variable,
      blobid,
      blob_lbl,
      mime
    )
  end

  state.workspace["Odometry"] = state.prv_variable

  nothing
end

