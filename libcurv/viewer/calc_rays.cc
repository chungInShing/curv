// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include "libcurv/traced_shape.h"
#include <libcurv/viewer/calc_rays.h>
#include <libcurv/die.h>
#include <CL/cl.h>
#include <memory>
#include <iostream>
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

RayCalcRetCode RayCalc::compileProgram(Traced_Shape& shape) {
    RayCalcRetCode result = RayCalcRetCode::OK;
    if (clkernel_ != nullptr) {
        clReleaseKernel(clkernel_);
        clkernel_ = nullptr;
    }
    if (clprog_ != nullptr) {
        clReleaseProgram(clprog_);
        clprog_ = nullptr;
    }
    //Compile program.
    cl_int err;
    const char* src[] = {shape.clprog_.c_str(), NULL};
    clprog_ = clCreateProgramWithSource(clContext_,
                                        1,
                                        src,
                                        NULL,
                                        &err
                                        );
    if (err != CL_SUCCESS) {
        std::cout << "Error creating program." << std::endl;
        print_opencl_results(err);
        result = RayCalcRetCode::COMPILE_ERROR;
    } else {
        std::cout << "Program created successfully." << std::endl;
        err = clBuildProgram(clprog_, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "Error building program." << std::endl;
            print_opencl_results(err);
            const int MAX_BUFFER_SIZE = 100000;
            char buffer[MAX_BUFFER_SIZE];
            size_t length = 0;
            clGetProgramBuildInfo(clprog_, device_id_, CL_PROGRAM_BUILD_LOG, sizeof(char) * MAX_BUFFER_SIZE, buffer, &length);
            std::cout << "Error buffer length is " << std::to_string(length) << std::endl;
            std::cout << buffer << std::endl;
            result = RayCalcRetCode::COMPILE_ERROR;
            std::cout << "Source openCL:" <<
                "-------------------" << std::endl << std::endl <<
                shape.clprog_ << "-------------------" << std::endl;
        } else {
            std::cout << "Program built successfully." << std::endl;
            clkernel_ = clCreateKernel(clprog_, "main", &err);
            if (!clkernel_ || err != CL_SUCCESS) {
                std::cout << "Error creating kernel." << std::endl;
                print_opencl_results(err);
                result = RayCalcRetCode::ERROR;
            }
        }
    }
    return result;
}

void RayCalc::setKernelArgs(const std::string& paramName, int index,
                            const Traced_Shape::VarType& paramType,
                            const bool isArray, const size_t size, void* memObj) {

    if (clprog_ != nullptr && clkernel_ != nullptr) {
        cl_int err = 0;
        err = clSetKernelArg(clkernel_, index, size, memObj);
        if (err != CL_SUCCESS) {
            std::cout << "Error setting parameter name: " << paramName << ", at index: " << index
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

RayCalcResult RayCalc::calculate(Traced_Shape& shape) {
    RayCalcResult result;
    shape.setInitialRays();
    bool finished = false;
    uint iterations = 1;
    cl_int err;
    do {
        //Tuple of parameters, with host memery obj.
        std::vector<std::tuple<
        decltype(shape.getKernelArgParams())::value_type,
        cl_mem>> buffers;
        for (auto param : shape.getKernelArgParams()) {
            cl_mem memObj = NULL;
            //Create memory object.
            memObj = clCreateBuffer(clContext_, std::get<6>(param) |
                                    CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                                    std::get<4>(param), std::get<5>(param), &err);
            if (memObj != NULL && err == CL_SUCCESS) {
                buffers.push_back(std::make_tuple(param, memObj));
                //Load kernel parameters.
                setKernelArgs(std::get<0>(param), std::get<1>(param), std::get<2>(param),
                              std::get<3>(param), sizeof(cl_mem), (void*)&memObj);
            } else {
                std::cout << "Error creating memory object for parameter " <<
                        std::get<0>(param) << ", index " <<
                        std::to_string(std::get<1>(param)) << ", data type " <<
                        std::to_string(std::get<2>(param)) << ", is array " <<
                        std::to_string(std::get<3>(param)) << ", buffer size" <<
                        std::to_string(std::get<4>(param)) << std::endl;
                die("Error creating memory object");
            }
        }
        //Load to queue.
        //Queue transfer from host to device.
        //for (auto e : buffers) {
        //    auto param = std::get<0>(e);
        //    auto memObj = std::get<1>(e);
        //    if (std::get<6>(param) == CL_MEM_READ_ONLY ||
        //            std::get<6>(param) == CL_MEM_READ_WRITE) {
        //        err = clEnqueueWriteBuffer(command_queue_, memObj, CL_TRUE, 0,
        //                std::get<4>(param), std::get<5>(param) , 0, NULL, NULL);
        //        if (err != CL_SUCCESS) {
        //            std::cout << "Error writing to buffer pameter " <<
        //                std::get<0>(param) << ", index " <<
        //                std::to_string(std::get<1>(param)) << ", data type " <<
        //                std::to_string(std::get<2>(param)) << ", is array " <<
        //                std::to_string(std::get<3>(param)) << ", buffer size" <<
        //                std::to_string(std::get<4>(param)) << std::endl;
        //        }
        //    }
        //}

        // Queue OpenCL kernel on the list
        size_t global_item_size = shape.getNumRays(); // Process the entire lists
        size_t local_item_size = shape.getNumRays();
        err = clEnqueueNDRangeKernel(command_queue_, clkernel_, 1, NULL,
                &global_item_size, &local_item_size, 0, NULL, NULL);
        //Queue transfer from device to host.
        for (auto e : buffers) {
            auto param = std::get<0>(e);
            auto memObj = std::get<1>(e);
            //if (std::get<6>(param) == CL_MEM_WRITE_ONLY ||
            //        std::get<6>(param) == CL_MEM_READ_WRITE) {
                err = clEnqueueReadBuffer(command_queue_, memObj, CL_TRUE, 0,
                        std::get<4>(param), std::get<5>(param) , 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    std::cout << "Error writing from buffer pameter " <<
                        std::get<0>(param) << ", index " <<
                        std::to_string(std::get<1>(param)) << ", data type " <<
                        std::to_string(std::get<2>(param)) << ", is array " <<
                        std::to_string(std::get<3>(param)) << ", buffer size" <<
                        std::to_string(std::get<4>(param)) << std::endl;
                }
            //}
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

    result.rays = shape.getResultRays();
    result.numInitialRays = shape.getNumRays();
    result.numHits = 0;
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

