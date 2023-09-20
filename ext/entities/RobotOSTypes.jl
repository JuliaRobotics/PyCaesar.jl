
# PyCaesar weakdeps extension structs

struct RosbagWriter end

mutable struct RosbagSubscriber{PYOBJ}
  bagfile::String
  channels::Vector{Symbol}
  callbacks::Dict{Symbol, Function}
  readers::Dict{Symbol,PYOBJ} # PyObject
  syncBuffer::Dict{Symbol,Tuple{DateTime, Int}} # t,ns,msgdata
  nextMsgChl::Symbol
  nextMsgTimestamp::Tuple{DateTime, Int}
  compression::Any
  # constructors
end