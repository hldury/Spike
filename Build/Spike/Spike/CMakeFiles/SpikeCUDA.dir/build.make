# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.3.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.3.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hakunahahannah/Documents/Projects/Spike

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hakunahahannah/Documents/Projects/Spike/Build

# Utility rule file for SpikeCUDA.

# Include the progress variables for this target.
include Spike/Spike/CMakeFiles/SpikeCUDA.dir/progress.make

Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/ActivityMonitor.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/RateActivityMonitor.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/SpikingActivityMonitor.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/CUDABackend.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/Memory.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/RandomStateManager.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/GeneratorInputSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/ImagePoissonInputSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/InputSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/LIFSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/Neurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/PatternedPoissonInputSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/PoissonInputSpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/SpikingNeurons.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/CustomSTDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/EvansSTDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/InhibitorySTDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/Plasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/STDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/WeightDependentSTDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/WeightNormSTDPPlasticity.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/ConductanceSpikingSynapses.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/CurrentSpikingSynapses.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/SpikingSynapses.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/Synapses.cu.o
Spike/Spike/CMakeFiles/SpikeCUDA: Spike/Spike/VoltageSpikingSynapses.cu.o
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking SpikeCUDA"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --red Linking\ CXX\ executable\ SpikeCUDA
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -dlink -Xcompiler '-fPIC' -o /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike/libSpikeCUDA.dlink.o ActivityMonitor.cu.o RateActivityMonitor.cu.o SpikingActivityMonitor.cu.o CUDABackend.cu.o Memory.cu.o RandomStateManager.cu.o GeneratorInputSpikingNeurons.cu.o ImagePoissonInputSpikingNeurons.cu.o InputSpikingNeurons.cu.o LIFSpikingNeurons.cu.o Neurons.cu.o PatternedPoissonInputSpikingNeurons.cu.o PoissonInputSpikingNeurons.cu.o SpikingNeurons.cu.o CustomSTDPPlasticity.cu.o EvansSTDPPlasticity.cu.o InhibitorySTDPPlasticity.cu.o Plasticity.cu.o STDPPlasticity.cu.o WeightDependentSTDPPlasticity.cu.o WeightNormSTDPPlasticity.cu.o ConductanceSpikingSynapses.cu.o CurrentSpikingSynapses.cu.o SpikingSynapses.cu.o Synapses.cu.o VoltageSpikingSynapses.cu.o -lcudadevrt
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -lib -o /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike/libSpikeCUDA.a ActivityMonitor.cu.o RateActivityMonitor.cu.o SpikingActivityMonitor.cu.o CUDABackend.cu.o Memory.cu.o RandomStateManager.cu.o GeneratorInputSpikingNeurons.cu.o ImagePoissonInputSpikingNeurons.cu.o InputSpikingNeurons.cu.o LIFSpikingNeurons.cu.o Neurons.cu.o PatternedPoissonInputSpikingNeurons.cu.o PoissonInputSpikingNeurons.cu.o SpikingNeurons.cu.o CustomSTDPPlasticity.cu.o EvansSTDPPlasticity.cu.o InhibitorySTDPPlasticity.cu.o Plasticity.cu.o STDPPlasticity.cu.o WeightDependentSTDPPlasticity.cu.o WeightNormSTDPPlasticity.cu.o ConductanceSpikingSynapses.cu.o CurrentSpikingSynapses.cu.o SpikingSynapses.cu.o Synapses.cu.o VoltageSpikingSynapses.cu.o /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike/libSpikeCUDA.dlink.o -lcudadevrt

Spike/Spike/ActivityMonitor.cu.o: ../Spike/Spike/Backend/CUDA/ActivityMonitor/ActivityMonitor.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/ActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/ActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/ActivityMonitor.cu -o ActivityMonitor.cu.o

Spike/Spike/RateActivityMonitor.cu.o: ../Spike/Spike/Backend/CUDA/ActivityMonitor/RateActivityMonitor.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/RateActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/RateActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/RateActivityMonitor.cu -o RateActivityMonitor.cu.o

Spike/Spike/SpikingActivityMonitor.cu.o: ../Spike/Spike/Backend/CUDA/ActivityMonitor/SpikingActivityMonitor.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/SpikingActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/SpikingActivityMonitor.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/ActivityMonitor/SpikingActivityMonitor.cu -o SpikingActivityMonitor.cu.o

Spike/Spike/CUDABackend.cu.o: ../Spike/Spike/Backend/CUDA/CUDABackend.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/CUDABackend.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/CUDABackend.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/CUDABackend.cu -o CUDABackend.cu.o

Spike/Spike/Memory.cu.o: ../Spike/Spike/Backend/CUDA/Helpers/Memory.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/Memory.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/Memory.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/Memory.cu -o Memory.cu.o

Spike/Spike/RandomStateManager.cu.o: ../Spike/Spike/Backend/CUDA/Helpers/RandomStateManager.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/RandomStateManager.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/RandomStateManager.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Helpers/RandomStateManager.cu -o RandomStateManager.cu.o

Spike/Spike/GeneratorInputSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.cu -o GeneratorInputSpikingNeurons.cu.o

Spike/Spike/ImagePoissonInputSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/ImagePoissonInputSpikingNeurons.cu -o ImagePoissonInputSpikingNeurons.cu.o

Spike/Spike/InputSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/InputSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/InputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/InputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/InputSpikingNeurons.cu -o InputSpikingNeurons.cu.o

Spike/Spike/LIFSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.cu -o LIFSpikingNeurons.cu.o

Spike/Spike/Neurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/Neurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/Neurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/Neurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/Neurons.cu -o Neurons.cu.o

Spike/Spike/PatternedPoissonInputSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/PatternedPoissonInputSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PatternedPoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PatternedPoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PatternedPoissonInputSpikingNeurons.cu -o PatternedPoissonInputSpikingNeurons.cu.o

Spike/Spike/PoissonInputSpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.cu -o PoissonInputSpikingNeurons.cu.o

Spike/Spike/SpikingNeurons.cu.o: ../Spike/Spike/Backend/CUDA/Neurons/SpikingNeurons.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/SpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/SpikingNeurons.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Neurons/SpikingNeurons.cu -o SpikingNeurons.cu.o

Spike/Spike/CustomSTDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/CustomSTDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/CustomSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/CustomSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/CustomSTDPPlasticity.cu -o CustomSTDPPlasticity.cu.o

Spike/Spike/EvansSTDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/EvansSTDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/EvansSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/EvansSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/EvansSTDPPlasticity.cu -o EvansSTDPPlasticity.cu.o

Spike/Spike/InhibitorySTDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/InhibitorySTDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/InhibitorySTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/InhibitorySTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/InhibitorySTDPPlasticity.cu -o InhibitorySTDPPlasticity.cu.o

Spike/Spike/Plasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/Plasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/Plasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/Plasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/Plasticity.cu -o Plasticity.cu.o

Spike/Spike/STDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/STDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/STDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/STDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/STDPPlasticity.cu -o STDPPlasticity.cu.o

Spike/Spike/WeightDependentSTDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_21) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightDependentSTDPPlasticity.cu -o WeightDependentSTDPPlasticity.cu.o

Spike/Spike/WeightNormSTDPPlasticity.cu.o: ../Spike/Spike/Backend/CUDA/Plasticity/WeightNormSTDPPlasticity.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_22) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightNormSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightNormSTDPPlasticity.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Plasticity/WeightNormSTDPPlasticity.cu -o WeightNormSTDPPlasticity.cu.o

Spike/Spike/ConductanceSpikingSynapses.cu.o: ../Spike/Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_23) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/ConductanceSpikingSynapses.cu -o ConductanceSpikingSynapses.cu.o

Spike/Spike/CurrentSpikingSynapses.cu.o: ../Spike/Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_24) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.cu -o CurrentSpikingSynapses.cu.o

Spike/Spike/SpikingSynapses.cu.o: ../Spike/Spike/Backend/CUDA/Synapses/SpikingSynapses.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_25) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/SpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/SpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/SpikingSynapses.cu -o SpikingSynapses.cu.o

Spike/Spike/Synapses.cu.o: ../Spike/Spike/Backend/CUDA/Synapses/Synapses.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_26) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/Synapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/Synapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/Synapses.cu -o Synapses.cu.o

Spike/Spike/VoltageSpikingSynapses.cu.o: ../Spike/Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/hakunahahannah/Documents/Projects/Spike/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_27) "Building /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/Cellar/cmake/3.3.2/bin/cmake -E cmake_echo_color --blue "Building NVCC Device object /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.cu"
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && /usr/local/cuda/bin/nvcc -std=c++11 -arch=sm_37 -I/Users/hakunahahannah/Documents/Projects/Spike/Spike -Xcompiler "'-fPIC'" -dc /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike/Backend/CUDA/Synapses/VoltageSpikingSynapses.cu -o VoltageSpikingSynapses.cu.o

SpikeCUDA: Spike/Spike/CMakeFiles/SpikeCUDA
SpikeCUDA: Spike/Spike/ActivityMonitor.cu.o
SpikeCUDA: Spike/Spike/RateActivityMonitor.cu.o
SpikeCUDA: Spike/Spike/SpikingActivityMonitor.cu.o
SpikeCUDA: Spike/Spike/CUDABackend.cu.o
SpikeCUDA: Spike/Spike/Memory.cu.o
SpikeCUDA: Spike/Spike/RandomStateManager.cu.o
SpikeCUDA: Spike/Spike/GeneratorInputSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/ImagePoissonInputSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/InputSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/LIFSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/Neurons.cu.o
SpikeCUDA: Spike/Spike/PatternedPoissonInputSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/PoissonInputSpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/SpikingNeurons.cu.o
SpikeCUDA: Spike/Spike/CustomSTDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/EvansSTDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/InhibitorySTDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/Plasticity.cu.o
SpikeCUDA: Spike/Spike/STDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/WeightDependentSTDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/WeightNormSTDPPlasticity.cu.o
SpikeCUDA: Spike/Spike/ConductanceSpikingSynapses.cu.o
SpikeCUDA: Spike/Spike/CurrentSpikingSynapses.cu.o
SpikeCUDA: Spike/Spike/SpikingSynapses.cu.o
SpikeCUDA: Spike/Spike/Synapses.cu.o
SpikeCUDA: Spike/Spike/VoltageSpikingSynapses.cu.o
SpikeCUDA: Spike/Spike/CMakeFiles/SpikeCUDA.dir/build.make

.PHONY : SpikeCUDA

# Rule to build all files generated by this target.
Spike/Spike/CMakeFiles/SpikeCUDA.dir/build: SpikeCUDA

.PHONY : Spike/Spike/CMakeFiles/SpikeCUDA.dir/build

Spike/Spike/CMakeFiles/SpikeCUDA.dir/clean:
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike && $(CMAKE_COMMAND) -P CMakeFiles/SpikeCUDA.dir/cmake_clean.cmake
.PHONY : Spike/Spike/CMakeFiles/SpikeCUDA.dir/clean

Spike/Spike/CMakeFiles/SpikeCUDA.dir/depend:
	cd /Users/hakunahahannah/Documents/Projects/Spike/Build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hakunahahannah/Documents/Projects/Spike /Users/hakunahahannah/Documents/Projects/Spike/Spike/Spike /Users/hakunahahannah/Documents/Projects/Spike/Build /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike /Users/hakunahahannah/Documents/Projects/Spike/Build/Spike/Spike/CMakeFiles/SpikeCUDA.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Spike/Spike/CMakeFiles/SpikeCUDA.dir/depend

