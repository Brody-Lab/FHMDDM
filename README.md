# setting up Julia
Log into `scotty`, the Princeton Neuroscience Institute's interactive system for developing software to run on computational clusters 
```
> ssh <netID@scotty.princeton.edu> 
```
Load a version of Julia
```
[<netID>scotty ~]$ module load julia\1.6.0
[<netID>scotty ~]$ julia
julia>
```
The availability of other versions of Julia on `scotty` can be checked by calling
```
[<netID>scotty ~]$ module avail julia
```
# installing the repository
In a Julia read-eval-print loop (REPL), enter the Pkg REPL by pressing ] from the Julia REPL. To get back to the Julia REPL, press backspace or ^C.
```
julia> ]
(v1.6) pkg> 
```
Add the `FHMDDM` repository
```
(v1.6) pkg> add https://github.com/Brody-Lab/FHMDDM.git
pkg> up
pkg> <backspace>
julia>
```
#  parsing data

# loading model
In a Julia script or read-eval-print loop (REPL)
```
julia> using FHMDDM
julia> datapath = ""
```