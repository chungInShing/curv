// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_TRACED_SHAPE_H
#define LIBCURV_TRACED_SHAPE_H

#include <glm/glm.hpp>
#include <libcurv/viewed_shape.h>
#include <string>
#include <CL/cl.h>
#include <optional>
#include <tuple>

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

    std::string clprog_;

    uint numRays_ = 0;
    bool finished_ = 0;
    std::vector<Ray> rays_;

    Traced_Shape() {};

    Traced_Shape(const Shape_Program& shape, const Render_Opts& opts);

    bool empty() { return clprog_.empty() || frag_.empty(); }

    //Returns kernel index from the specific variable name and type. Returns -1 is not found.
    uint getKernelVarIndex(const std::string& varName, const VarType type, const bool isArray);

    //Set initial vector of rays to the data structure.
    void setInitialRays(const std::vector<Ray>& inputRays);
    void setInitialRays();

    //Get number of rays.
    uint getNumRays() { return numRays_; }
    //Get result rays.
    const std::vector<Ray> getResultRays() { return rays_; };
    //Propagate ray calculation, returns true if calculation is finished.
    bool propagate();
    //Get kernel args parameters.
    // Returns a tuple of parameter name, index, data type, is array, data array size, pointer to param array, openCl buffer flags.
    std::vector<std::tuple<std::string, int, Traced_Shape::VarType, bool, size_t, void*, cl_mem_flags>>
        getKernelArgParams();

    //Data storage.
    std::map<std::string, MemDataAttr> argsData_;

    //Getters and setters.
    std::optional<MemDataAttr> getData(const std::string& param);

};

} //namespace
#endif //LIBCURV_TRACED_SHAPE_H
