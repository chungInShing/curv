// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include "libcurv/traced_shape.h"
#include <libcurv/viewer/calc_rays.h>
#include <libcurv/die.h>
#include <CL/cl.h>
#include <memory>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>

namespace curv {
namespace viewer {

void print_opencl_results(cl_int code) {
    switch (code) {
        case CL_SUCCESS:
            std::cout << "CL_SUCCESS" << std::endl;
            break;
        case CL_INVALID_VALUE:
            std::cout << "CL_INVALID_VALUE" << std::endl;
            break;
        case CL_OUT_OF_HOST_MEMORY:
            std::cout << "CL_OUT_OF_HOST_MEMORY" << std::endl;
            break;
        case CL_INVALID_DEVICE:
            std::cout << "CL_INVALID_DEVICE" << std::endl;
            break;
        case CL_INVALID_BUILD_OPTIONS:
            std::cout << "CL_INVALID_BUILD_OPTIONS" << std::endl;
            break;
        case CL_BUILD_PROGRAM_FAILURE:
            std::cout << "CL_BUILD_PROGRAM_FAILURE" << std::endl;
            break;
        case CL_INVALID_KERNEL_NAME:
            std::cout << "CL_INVALID_KERNEL_NAME" << std::endl;
            break;
        default:
            std::cout << "Unknown code: " << std::to_string(code) << std::endl;
    }
}

RayCalc::RayCalc() {
    //Constructor.
}

RayCalc::RayCalc(RayCalc::Parameters param) : param_(param) {
    //Constructor.
}

RayCalc::~RayCalc() {
}

RayCalcRetCode RayCalc::init() {
    RayCalcRetCode result = RayCalcRetCode::OK;
    if (!isInit()) {
        if (initCL()) {
        } else {
            result = RayCalcRetCode::INIT_ERROR;
        }
    }
    if (result == RayCalcRetCode::OK) initialized_ = true;
    return result;
}

void RayCalc::close() {
    if (isInit()) {
        closeCL();
        initialized_ = false;
    }
}

std::optional<cl_program>  RayCalc::compileProgram(const std::string& source, RayCalcRetCode& code) {
    cl_program prog = nullptr;
    //Compile program.
    cl_int err;
    const char* src[] = {source.c_str(), NULL};
    prog = clCreateProgramWithSource(clContext_,
                                        1,
                                        src,
                                        NULL,
                                        &err
                                        );
    if (err != CL_SUCCESS) {
        std::cout << "Error creating program." << std::endl;
        print_opencl_results(err);
        code = RayCalcRetCode::COMPILE_ERROR;
    } else {
        std::cout << "Program created successfully." << std::endl;
        err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "Error building program." << std::endl;
            print_opencl_results(err);
            const int MAX_BUFFER_SIZE = 100000;
            char buffer[MAX_BUFFER_SIZE];
            size_t length = 0;
            clGetProgramBuildInfo(prog, device_id_, CL_PROGRAM_BUILD_LOG, sizeof(char) * MAX_BUFFER_SIZE, buffer, &length);
            std::cout << "Error buffer length is " << std::to_string(length) << std::endl;
            std::cout << buffer << std::endl;
            code = RayCalcRetCode::COMPILE_ERROR;
            std::cout << "Source openCL:" <<
                "-------------------" << std::endl << std::endl <<
                source << "-------------------" << std::endl;
        } else {
            std::cout << "Program built successfully." << std::endl;
            code = RayCalcRetCode::OK;
        }
    }
    return code == RayCalcRetCode::OK && prog != nullptr ?
             std::optional<cl_program>(prog) : std::nullopt;
}

std::optional<cl_kernel> RayCalc::genKernel(cl_program prog, const std::string& kernelName, RayCalcRetCode& code) {
    cl_int err;
    cl_kernel kernel = nullptr;
    if (prog == nullptr || kernelName.empty()) {
        std::cout << "Program or kernel name is null" << std::endl;
        code = RayCalcRetCode::INPUT_ERROR;
    } else {
        kernel = clCreateKernel(prog, kernelName.c_str(), &err);
        if (!kernel || err != CL_SUCCESS) {
            std::cout << "Error creating kernel." << std::endl;
            print_opencl_results(err);
            code = RayCalcRetCode::ERROR;
        } else {
            std::cout << "Kernel create successfully." << std::endl;
            code = RayCalcRetCode::OK;
        }
    }
    return code == RayCalcRetCode::OK && kernel != nullptr ?
            std::optional<cl_kernel>(kernel) : std::nullopt;
}

void RayCalc::setKernelArgs(cl_kernel& kernel, int index,
                            const Traced_Shape::VarType& paramType,
                            const bool isArray, const size_t size, void* memObj) {

    if (kernel != nullptr) {
        cl_int err = 0;
        err = clSetKernelArg(kernel, index, size, memObj);
        if (err != CL_SUCCESS) {
            std::cout << "Error setting parameter index: " << index
                      << ", with type: " << std::to_string(paramType)
                      << ", and is arrray: " << std::to_string(isArray) << ", with size: "
                      << size << ", at address: " << memObj << ", return value"
                      << std::to_string(err) << std::endl;
            die("Error setting kernel parameter.");
        }
    } else {
        die("OpenCL program or kernel not found");
    }
}

RayCalcRetCode RayCalc::setParameters(Traced_Shape& shape) {
    RayCalcRetCode result = RayCalcRetCode::OK;
    //Parse parameters.

    //Set parameters to kernel.
    //Refraction index ratio(float).
    return result;
}
std::optional<cl_mem> RayCalc::createAndLoadBuffer(cl_kernel kernel, const Traced_Shape::KernelParam& param) {
    cl_int err;
    cl_mem memObj = NULL;
    //Create memory object.
    memObj = clCreateBuffer(clContext_, param.bufferFlags_ |
                            CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                            param.bufferSize_, param.bufferPtr_, &err);
    if (memObj != NULL && err == CL_SUCCESS) {
        //Load kernel parameters.
        setKernelArgs(kernel, param.index_, param.varType_,
                      param.isArray_, sizeof(cl_mem), (void*)&memObj);
    } else {
        std::cout << "Error creating memory object for parameter " <<
                param.name_ << ", index " <<
                std::to_string(param.index_) << ", data type " <<
                std::to_string(param.varType_) << ", is array " <<
                std::to_string(param.isArray_) << ", buffer size" <<
                std::to_string(param.bufferSize_) << std::endl;
    }
    return (memObj != NULL && err == CL_SUCCESS) ?
        std::optional<cl_mem>(memObj) : std::nullopt;

}

cl_int RayCalc::runKernel(cl_kernel kernel, size_t* global_size, size_t* local_size) {
    return clEnqueueNDRangeKernel(command_queue_, kernel, 1, NULL,
                global_size, local_size, 0, NULL, NULL);
}

cl_int RayCalc::readBack(cl_mem memObj, const Traced_Shape::KernelParam& param) {
    //if (param.bufferFlags_ == CL_MEM_WRITE_ONLY ||
    //        param.bufferFlags_ == CL_MEM_READ_WRITE) {
        cl_int err = clEnqueueReadBuffer(command_queue_, memObj, CL_TRUE, 0,
                param.bufferSize_, param.bufferPtr_ , 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "Error writing from buffer pameter " <<
                param.name_ << ", index " <<
                std::to_string(param.index_) << ", data type " <<
                std::to_string(param.varType_) << ", is array " <<
                std::to_string(param.isArray_) << ", buffer size" <<
                std::to_string(param.bufferSize_) << std::endl;
        }
    //}
    return err;
}

RayCalcResult RayCalc::calculate(Traced_Shape& shape) {
    RayCalcResult result;
    bool finished = false;
    uint iterations = 1;
    cl_int err = 0;
    RayCalcRetCode code;
    shape.setInitialRays();
    if (shape.getNumRays() > 0) {
        //Ray initialization (if exists)
        if (shape.calc_init_rays_) {
            if (auto initprog = compileProgram(shape.clinitprog_, code)) {
                if(auto kernel = genKernel(initprog.value(), shape.getInitRayKernelName(), code)) {
                    //Tuple of parameters, with host memery obj.
                    std::vector<std::tuple<
                    Traced_Shape::KernelParam,
                    cl_mem>> buffers;
                    for (auto param : shape.getRayInitArgParams()) {
                         if (auto memObj = createAndLoadBuffer(kernel.value(), param)) {
                            buffers.push_back(std::make_tuple(param, memObj.value()));
                         } else {
                             die("Error creating and loading buffer.");
                         }
                    }
                    // Queue OpenCL kernel on the list
                    size_t global_item_size = shape.getNumRays(); // Process the entire lists
                    size_t local_item_size = shape.getNumRays();
                    err &= runKernel(kernel.value(), &global_item_size, &local_item_size);
                    //Queue transfer from device to host.
                    for (auto e : buffers) {
                        auto param = std::get<0>(e);
                        auto memObj = std::get<1>(e);
                        err &= readBack(memObj, param);
                    }
                    //Let the calculation finish.
                    //clFlush(command_queue_);
                    clFinish(command_queue_);
                    //Release buffers.
                    for (auto b : buffers)
                        if (std::get<1>(b) != nullptr) {
                            clReleaseMemObject(std::get<1>(b));
                        }
                } else {
                    std::cout << "Ray initialization kernel failed to build." << std::endl;
                }
            } else {
                std::cout << "Ray initialization program failed to build." << std::endl;
            }
        }
        //Ray propagation.
        if (auto prog = compileProgram(shape.clprog_, code)) {
            if (auto kernel = genKernel(prog.value(), shape.getRayCalcKernelName(), code)) {
                do {
                    //Tuple of parameters, with host memery obj.
                    std::vector<std::tuple<
                    Traced_Shape::KernelParam,
                    cl_mem>> buffers;
                    for (auto param : shape.getKernelArgParams()) {
                         if (auto memObj = createAndLoadBuffer(kernel.value(), param)) {
                            buffers.push_back(std::make_tuple(param, memObj.value()));
                         } else {
                             die("Error creating and loading buffer.");
                         }
                    }
                    // Queue OpenCL kernel on the list
                    size_t global_item_size = shape.getNumRays(); // Process the entire lists
                    size_t local_item_size = shape.getNumRays();
                    err &= runKernel(kernel.value(), &global_item_size, &local_item_size);
                    //Queue transfer from device to host.
                    for (auto e : buffers) {
                        auto param = std::get<0>(e);
                        auto memObj = std::get<1>(e);
                        err &= readBack(memObj, param);
                    }
                    //Let the calculation finish.
                    //clFlush(command_queue_);
                    clFinish(command_queue_);
                    //Release buffers.
                    for (auto b : buffers)
                        if (std::get<1>(b) != nullptr) {
                            clReleaseMemObject(std::get<1>(b));
                        }
                    //Get result.
                    finished = shape.propagate();
                    iterations++;
                } while (!finished && iterations < param_.maxIter);
            } else {
                //die ("Error creating kernel");
                std::cout << "Error creating kernel" << std::endl;
            }
        } else {
            //die ("Error creating program");
            std::cout << "Error creating program" << std::endl;
        }
        result.rays = shape.getResultRays();
        result.numInitialRays = shape.getNumRays();
        result.numHits = 0;
    }
    if (err != CL_SUCCESS) {
        result.returnCode = RayCalcRetCode::ERROR;
    } else {
        result.returnCode = RayCalcRetCode::OK;
    }
    return result;
}

bool RayCalc::initCL() {
    bool success = false;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id_, &ret_num_devices);
    clContext_ = clCreateContext( NULL, 1, &device_id_, NULL, NULL, &ret);
    command_queue_ = clCreateCommandQueue(clContext_, device_id_,
                                                        0, &ret);
    if (clContext_ != NULL && command_queue_ != NULL) {
        success = true;
        std::cout << "initCL successful." << std::endl;
    } else {
        std::cout << "initCL failed." << std::endl;
        closeCL();
    }
    return success;
}

void RayCalc::setup() {

}

void RayCalc::closeCL() {
    if (command_queue_ != NULL) {
        clFlush(command_queue_);
        clFinish(command_queue_);
        clReleaseCommandQueue(command_queue_);
    }
    if (clContext_ != NULL) {
        clReleaseContext(clContext_);
        clContext_ = NULL;
    }
}

}
}

