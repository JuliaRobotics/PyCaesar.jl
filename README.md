# PyCaesar.jl

| Stable | Dev | Coverage | Docs |
|--------|-----|----------|------|
| [![version][pycjl-ver-img]][pycjl-releases] | [![CI][pycjl-ci-dev-img]][pycjl-ci-dev-url] | [![codecov.io][pycjl-cov-img]][pycjl-cov-url] | [![docs][cjl-docs-img]][cjl-docs-url] <br> [![][caesar-slack-badge]][caesar-slack] |

## Dependencies

The following packages should be available in the Python environment used by PyCall.jl

```
rospy
opencv-python # i.e. cv2
open3d
Rosbags
```

> 23Q3: We had issues in using Conda.jl to install these dependecies in a new environment, specifically conda package compliance with newer versions of Python.  E.g. open3d was not available via conda on Python 3.10 at the time of writing.  Please open an issue if further clarification is needed.

## Introduction

Caesar.jl extensions using Python.  See common [Caesar.jl][cjl-docs-url] Docs for details.


[pycjl-url]: http://www.github.com/JuliaRobotics/PyCaesar.jl
[pycjl-cov-img]: https://codecov.io/github/JuliaRobotics/PyCaesar.jl/coverage.svg?branch=develop
[pycjl-cov-url]: https://codecov.io/github/JuliaRobotics/PyCaesar.jl?branch=develop
[pycjl-ci-dev-img]: https://github.com/JuliaRobotics/PyCaesar.jl/actions/workflows/CI.yml/badge.svg
[pycjl-ci-dev-url]: https://github.com/JuliaRobotics/PyCaesar.jl/actions/workflows/CI.yml
[pycjl-ver-img]: https://juliahub.com/docs/PyCaesar/version.svg
[pycjl-milestones]: https://github.com/JuliaRobotics/PyCaesar.jl/milestones
[pycjl-releases]: https://github.com/JuliaRobotics/PyCaesar.jl/releases

[cjl-url]: https://github.com/JuliaRobotics/Caesar.jl
[cjl-docs-img]: https://img.shields.io/badge/docs-latest-blue.svg
[cjl-docs-url]: http://juliarobotics.github.io/Caesar.jl/latest/
[caesar-slack-badge]: https://img.shields.io/badge/Caesarjl-Slack-green.svg?style=popout
[caesar-slack]: https://join.slack.com/t/caesarjl/shared_invite/zt-ucs06bwg-y2tEbddwX1vR18MASnOLsw