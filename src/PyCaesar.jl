module PyCaesar

using PyCall
using Caesar
using Dates
using DocStringExtensions
using Pkg
using UUIDs
# using Unmarshal



# weakdeps exports
export RosbagWriter, RosbagSubscriber

"""
    version

Return PyCaesar's `::VersionNumber` when user calls internal function `PyCaesar.version()`.

Notes
- Not exported since this is member name is likely to occur in many packages. 
"""
version() = Pkg.dependencies()[UUID("5de271da-f4c9-48db-ba43-272b66d09ab8")].version


# special includes for weakdeps structs and members
include("../ext/entities/RobotOSTypes.jl")
include("../ext/services/WeakdepsPrototypes.jl")


end # module PyCaesar
