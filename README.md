# FSharp_MLIR

## Requirements

### Platform Support

Currently only Linux development is supported. While other platforms may work, they are not tested and may require
additional setup.
If you are feeling adventurous, you can try to change the build system following
the [MLIR documentation](https://mlir.llvm.org/getting_started/).

If you are using Windows, you can use the Windows Subsystem for Linux (WSL) to build and run the project.
We recommend using WSL2 with a recent Ubuntu distro or Debian if you want a more minimal experience.

Using Debian will probably require setup of more dependencies, but it should be possible to build the project.
Check the output of the build system for missing dependencies and install them as needed.

### Compiler Support

You will need a compiler that supports C++20 (no module support required) and CMake 3.20 or newer.

Since this project uses the LLVM project extensively we recommend using the Clang compiler to build the project since it
is the most tested compiler with LLVM, and also the only one that we can guarantee will work.

This project was developed using Clang 18.
Use the scripts below to install the required dependencies.

## Building

### Dependencies

FSharp MLIR requires the following dependencies to be installed on the system:
```bash
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" \
sudo apt install default-jdk \
sudo apt install ccache \
sudo apt install cmake \
sudo apt install build-essential \
sudo apt install lld \
sudo apt install ninja-build
```

### Building the Project

Clone the project into a directory of your choice and build the project using the following commands:

```bash
git clone https://github.com/LaFoster00/FSharp_MLIR.git --recursive --progress --shallow-submodules \
cd FSharp_MLIR \
mkdir build \
cd build \
cmake -G Ninja .. \
ninja
```

If you want to only build the FSharpCompilerApp target use the following command:

```bash
ninja FSharpCompilerApp
```

Configuring the project the first time will take a while since it will build the LLVM project. It will also download a
couple of additional dependencies such as boost which will also take some time.

## Running the Compiler
To run the compiler use the following command:

```bash
./compiler/app/FSharpCompilerApp -emit=exe -o <path-to-exe-outpout> <path-to-fsharp-source-file>
```

Possible options are:
```bash
./compiler/app/FSharpCompilerApp --help
OVERVIEW: fsharp compiler

USAGE: FSharpCompilerApp [options] <input fsharp file>

OPTIONS:

FSharp Compiler Options:
Specific options for FSharp compiler

  --emit=<value> - Select the kind of output desired
    =st          -   output the ST dump
    =ast         -   output the AST dump
    =mlir        -   output the MLIR dump
    =mlir-typed  -   output the MLIR dump after type inference
    =mlir-arith  -   output the MLIR dump after arith lowering
    =mlir-func   -   output the MLIR dump after func lowering
    =mlir-buff   -   output the MLIR dump after bufferization
    =mlir-llvm   -   output the MLIR dump after llvm lowering
    =llvm        -   output the LLVM IR dump
    =jit         -   JIT the code and run it by executing the input script
    =exe         -   Emit an executable from the input script
  -o <filename>  - Specify the output filename for the executable
  --opt          - Enable optimizations
  -x=<value>     - Decided the kind of output desired
    =fsharp      -   load the input file as a FSharp source.
    =mlir        -   load the input file as an MLIR file

Generic Options:

  --help         - Display available options (--help-hidden for more)
  --help-list    - Display list of available options (--help-list-hidden for more)
  --version      - Display the version of this program

```
