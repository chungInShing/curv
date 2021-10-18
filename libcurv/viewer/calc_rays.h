// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_VIEWER_CALC_RAYS_H
#define LIBCURV_VIEWER_CALC_RAYS_H
#include <libcurv/traced_shape.h>
#include <CL/cl.h>
#include <vector>
#include <tuple>
#include <map>
#include <memory>

namespace curv {
namespace viewer {

enum RayCalcRetCode {OK, ERROR, INPUT_ERROR, COMPILE_ERROR, INIT_ERROR};

struct RayCalcResult
{
    RayCalcRetCode returnCode = RayCalcRetCode::OK;
    std::vector<Ray> rays;
    int numInitialRays = 0;
    int numHits = 0;
};

struct RayCalc
{
//Parameters.
    struct Parameters { uint maxIter = 1000;};
/*--- Public functions ---*/
    RayCalc();
    RayCalc(Parameters param);
    ~RayCalc();
    RayCalcRetCode init();
    void close();
    std::optional<cl_program> compileProgram(const std::string& source, RayCalcRetCode& code);
    std::optional<cl_kernel> genKernel(cl_program prog, const std::string& kernelName, RayCalcRetCode& code);
    RayCalcRetCode setParameters(Traced_Shape& shape);
    RayCalcRetCode calcInitRays(Traced_Shape& shape);
    RayCalcRetCode calcPropagateRays(Traced_Shape& shape);
    RayCalcResult calculate(Traced_Shape& shape);
    bool isInit() { return initialized_; }
//OpenCL instances.
    cl_context clContext_ = NULL;
    cl_command_queue command_queue_ = NULL;
    cl_device_id device_id_ = NULL;
//OpenCL programs.
    Parameters param_;
/*--- INTERNAL STATE ---*/
    bool error_=false;
    bool initialized_=false;
// INTERNAL FUNCTIONS
    bool initCL();
    void setup();
    void closeCL();
    void setKernelArgs(cl_kernel& kernel, int index,
            const Traced_Shape::VarType& paramType, const bool isArray,
            const size_t size, void* memObj);
    std::optional<cl_mem> createAndLoadBuffer(cl_kernel kernel, const Traced_Shape::KernelParam& param);
    cl_int runKernel(cl_kernel kernel, size_t* global_size, size_t* local_size);
    cl_int readBack(cl_mem memObj, const Traced_Shape::KernelParam& param);


};
} //viewer
} //curv
#endif //header guard
