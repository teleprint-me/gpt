# Building GPT

This document explains how to build the `gpt` project with CMake and GDB for both Debug and Release configurations.

## Prerequisites

1. Install required development packages, libraries, and headers (e.g., libc++ or libstdc++)
2. Ensure you have a working installation of CMake
3. Create a build directory: `mkdir build`

## Build Process

### Debug Configuration

To build the project in debug mode with symbol information for GDB, follow these steps:

1. Clean any existing build artifacts (optional):

   ```sh
   rm -rf build
   mkdir build
   ```

2. Configure CMake to use a Debug build type and generate the necessary files:

   ```sh
   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   ```

3. Build the project using `cmake`'s `--build` command, specifying the number of jobs (e.g., 8):

   ```sh
   cmake --build build -j 8
   ```

4. Run GDB with the generated debug executable:

   ```sh
   gdb build/tokenizer
   ```

### Release Configuration

To build the project in release mode, follow these steps:

1. Clean any existing build artifacts (optional):

   ```sh
   rm -rf build
   mkdir build
   ```

2. Configure CMake to use a Release build type and generate the necessary files:

   ```sh
   cmake -B build
   ```

3. Build the project using `cmake`'s `--build` command, specifying the number of jobs (e.g., 8):

   ```sh
   cmake --build build -j 8
   ```

4. Run the generated release executable:

   ```sh
   ./bin/tokenizer
   ```

## Troubleshooting

If you encounter any issues during the build process, make sure to check for compiler warnings and investigate their causes.

## References

* CMake documentation: <http://www.cmake.org>
* GDB documentation: <http://www.gnu.org/software/gdb/documentation/>
