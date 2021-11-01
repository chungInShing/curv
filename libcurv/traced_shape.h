// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_TRACED_SHAPE_H
#define LIBCURV_TRACED_SHAPE_H

#include <cstddef>
#include <glm/glm.hpp>
#include <libcurv/viewed_shape.h>
#include <string>
#include <CL/cl.h>
#include <optional>
#include <tuple>
#include <libcurv/rays.h>


namespace curv {

struct Ray
{
    glm::vec3 pos; //Origin of ray.
    glm::vec3 dir; //Direction of ray.
    glm::vec4 colour; //Colour of ray.
    float refractIndRatio; //Ratio of the index of refraction upon hitting solid.
};


struct Traced_Shape : Viewed_Shape
{


    enum VarType {BOOL, INT, UINT, FLOAT, FLOAT2, FLOAT3, FLOAT4, UNKNOWN};

    enum ParamSet {RAY_INIT, KERNEL};

    struct MemDataAttr {
        std::shared_ptr<void> data_ = nullptr;
        std::string name_;
        VarType dataType_;
        size_t size_ = 0;
        MemDataAttr(const std::shared_ptr<void> data, const std::string& name, VarType dataType, size_t size)
            : data_(data), name_(name), dataType_(dataType), size_(size) {}
        MemDataAttr(const MemDataAttr& attr)
            : data_(attr.data_), name_(attr.name_), dataType_(attr.dataType_), size_(attr.size_) {}
        MemDataAttr() : data_(nullptr), name_(""), dataType_(VarType::UNKNOWN), size_(0) {}
    };

    struct KernelParam {
        std::string name_;
        int index_;
        VarType varType_;
        bool isArray_;
        size_t bufferSize_;
        void* bufferPtr_;
        cl_mem_flags bufferFlags_;
        KernelParam(const std::string& name, int index, VarType varType, bool isArray, size_t bufferSize, void* bufferPtr, cl_mem_flags bufferFlags) : name_(name), index_(index), varType_(varType), isArray_(isArray), bufferSize_(bufferSize), bufferPtr_(bufferPtr), bufferFlags_(bufferFlags) {}
    };


    std::string clprog_, clinitprog_;

    std::tuple<unsigned int, unsigned int, unsigned int> numRays_;
    bool finished_ = 0;
    bool calc_init_rays_ = 0;
    std::vector<Ray> rays_;

    Traced_Shape() {};

    Traced_Shape(const Shape_Program& shape, const Render_Opts& opts);

    Traced_Shape(const Shape_Program& shape, const Rays_Program& rays, const Render_Opts& opts);

    bool empty() { return clprog_.empty() || frag_.empty(); }

    //Returns kernel index from the specific variable name and type. Returns -1 is not found.
    uint getVarIndex(const std::vector<std::tuple<std::string, Traced_Shape::VarType, bool, cl_mem_flags>>& paramSet, const std::string& varName, const VarType type, const bool isArray);

    //Set initial vector of rays to the data structure.
    void setInitialRays(const std::vector<Ray>& inputRays);
    void setInitialRays();

    void setInitBuffers();
    void setInitBuffers(std::tuple<unsigned int, unsigned int, unsigned int> numRays);

    //Get number of rays.
    unsigned int getNumRays() { return std::get<0>(numRays_) * std::get<1>(numRays_) * std::get<2>(numRays_); }
    //Get result rays.
    const std::vector<Ray> getResultRays() { return rays_; };
    //Propagate ray calculation, returns true if calculation is finished.
    bool propagate();
    //Get kernel args parameters.
    // Returns a tuple of parameter name, index, data type, is array, data array size, pointer to param array, openCl buffer flags.
    //std::vector<std::tuple<std::string, int, Traced_Shape::VarType, bool, size_t, void*, cl_mem_flags>>
    std::vector<KernelParam> getKernelArgParams();
    //Get ray init arg parameters.
    std::vector<KernelParam> getRayInitArgParams();
    //General method to get all parameters.
    std::vector<KernelParam> getArgParams(ParamSet set);

    //Data storage.
    std::map<std::string, MemDataAttr> argsData_;

    //Getters and setters.
    std::optional<MemDataAttr> getData(const std::string& param);

    std::string getRayCalcKernelName();

    std::string getInitRayKernelName();

};

} //namespace
#endif //LIBCURV_TRACED_SHAPE_H
