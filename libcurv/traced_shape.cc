// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include "libcurv/shape.h"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <libcurv/traced_shape.h>
#include <libcurv/glsl.h>
#include <libcurv/die.h>
#include <memory>
#include <tuple>
#include <iostream>


#include <libcurv/context.h>
#include <libcurv/function.h>
#include <libcurv/sc_compiler.h>

namespace curv {

#ifdef OPENCL_TEST_KERNEL
static const char* TEST_KERNEL =
        "__kernel void main(__global int* message) {\n"
        "    int gid = get_global_id(0);\n"
        "    message[gid] += gid;\n"
        "}\n"
        "\n";
#endif

static const char* DEFAULT_HEADER = "#define vec2 float2\n"
                                    "#define vec3 float3\n"
                                    "#define vec4 float4\n"
                                    "#ifdef abs\n"
                                    "#undef abs\n"
                                    "#endif\n"
                                    "#define abs fabs\n"
                                    "#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME\n"
                                    "#define float2(...) GET_MACRO(__VA_ARGS__,float2_4, float2_3, float2_2, float2_1)(__VA_ARGS__)\n"
                                    "#define float3(...) GET_MACRO(__VA_ARGS__,float3_4, float3_3, float3_2, float3_1)(__VA_ARGS__)\n"
                                    "#define float4(...) GET_MACRO(__VA_ARGS__,float4_4, float4_3, float4_2, float4_1)(__VA_ARGS__)\n"
                                    "#define float2_4(X, Y, Z, W) ((float2)((X), (Y)))\n"
                                    "#define float2_3(X, Y, Z) ((float2)((X), (Y)))\n"
                                    "#define float2_2(X, Y) ((float2)((X), (Y)))\n"
                                    "#define float2_1(X) ((float2)((X), (X)))\n"
                                    "#define float3_4(X, Y, Z, W) ((float3)((X), (Y), (Z)))\n"
                                    "#define float3_3(X, Y, Z) ((float3)((X), (Y), (Z)))\n"
                                    "#define float3_2(X, Y) ((float3)((X), (Y), (Y)))\n"
                                    "#define float3_1(X) ((float3)((X), (X), (X)))\n"
                                    "#define float4_4(X, Y, Z, W) ((float4)((X), (Y), (Z), (W)))\n"
                                    "#define float4_3(X, Y, Z) ((float4)((X), (Y), (Z), (Z)))\n"
                                    "#define float4_2(X, Y) ((float4)((X), (Y), (Y), (Y)))\n"
                                    "#define float4_1(X) ((float4)((X), (X), (X), (X)))\n"
                                    "#define const __constant\n"
                                    "#define in __constant\n"
                                    "#define __const_global\n"
                                    "\n";

static const char* DEFAULT_REFLECT =
                    "vec3 reflect(float3 rd, float3 nor) {\n"
                    "    return rd - 2.0 * dot(nor, rd) * nor;\n"
                    "}\n"
                    "\n";

static const char* DEFAULT_REFRACTION =
                    "vec3 refract(float3 rd, float3 nor, float ind) {\n"
                    "    float k = 1.0 - ind * ind * (1.0 - dot(nor, rd) * dot(nor, rd));\n"
                    "    if (k < 0.0) {\n"
                    "        return float3(0.0);\n"
                    "    } else {\n"
                    "        return ind * rd - (ind * dot(nor, rd) + sqrt(k)) * nor;\n"
                    "    }\n"
                    "}\n"
                    "\n";

static const char* DEFAULT_IS_REFRACTION =
                    "bool isRefraction(float3 rd, float3 nor, float ind) {\n"
                    "    return isgreaterequal(1.0 - ind * ind * (1.0 - dot(nor, rd) * dot(nor, rd)), 0.0);\n"
                    "}\n"
                    "\n";

// Following code is based on code fragments written by Inigo Quilez,
// with The MIT Licence.
//    Copyright 2013 Inigo Quilez
static const char* DEFAULT_CAST_RAY =
       "// ray marching. ro is ray origin, rd is ray direction (unit vector).\n"
       "// result is (t,r,g,b), where\n"
       "//  * t is the distance that we marched,\n"
       "//  * r,g,b is the colour of the distance field at the point we ended up at.\n"
       "//    (-1,-1,-1) means no object was hit.\n"
       "vec4 castRay( float3 ro, float3 rd, float time )\n"
       "{\n"
       "    float tmin = 0.001;\n" // was 1.0
       "    float tmax = ray_max_depth;\n"
       "   \n"
       // TODO: implement bounding volume. If I remove the 'if(t>tmax)break'
       // check, then `tetrahedron` breaks. The hard coded tmax=200 fails for
       // some models.
       //"#if 0\n"
       //"    // bounding volume\n"
       //"    float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );\n"
       //"    float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );\n"
       //"                                                 else           tmax = min( tmax, tp2 ); }\n"
       //"#endif\n"
       //"    \n"
       "    float t = tmin;\n"
       "    float3 c = (float3)(-1.0,-1.0,-1.0);\n"
       "    for (int i=0; i<ray_max_iter; i++) {\n"
       "        float precis = 0.0005*t;\n"
       "        float4 p = (float4)(ro+rd*t,time);\n"
       "        float d = dist(p);\n"
       "        if (fabs(d) < precis) {\n"
       "            c = colour(p);\n"
       "            break;\n"
       "        }\n"
       "        t += fabs(d);\n"
       "        if (t > tmax) break;\n"
       "    }\n"
       "    return (float4)( t, c );\n"
       "}\n";

static const char* DEFAULT_CALC_NORMAL =
       "float3 calcNormal( float3 pos, float time )\n"
       "{\n"
       "    float2 e = (float2)(1.0,-1.0)*0.5773*0.0005;\n"
       "    float3 e1 = (float3)(e.x, e.y, e.y);\n"
       "    float3 e2 = (float3)(e.y, e.y, e.x);\n"
       "    float3 e3 = (float3)(e.y, e.x, e.y);\n"
       "    float3 e4 = (float3)(e.x, e.x, e.x);\n"
       "    return normalize( e1*dist( (float4)(pos + e1,time) ) + \n"
       "                      e2*dist( (float4)(pos + e2,time) ) + \n"
       "                      e3*dist( (float4)(pos + e3,time) ) + \n"
       "                      e4*dist( (float4)(pos + e4,time) ) );\n"
//       "    return normalize( e.xyy*dist( (float4)(pos + e.xyy,time) ) + \n"
//       "                      e.yyx*dist( (float4)(pos + e.yyx,time) ) + \n"
//       "                      e.yxy*dist( (float4)(pos + e.yxy,time) ) + \n"
//       "                      e.xxx*dist( (float4)(pos + e.xxx,time) ) );\n"
       //"    /*\n"
       //"    vec3 eps = vec3( 0.0005, 0.0, 0.0 );\n"
       //"    vec3 nor = vec3(\n"
       //"        dist(pos+eps.xyy) - dist(pos-eps.xyy),\n"
       //"        dist(pos+eps.yxy) - dist(pos-eps.yxy),\n"
       //"        dist(pos+eps.yyx) - dist(pos-eps.yyx) );\n"
       //"    return normalize(nor);\n"
       //"    */\n"
       "}\n";

static const std::vector<std::tuple<std::string, Traced_Shape::VarType, bool, cl_mem_flags>>
    DEFAULT_KERNEL_PARAMETER=
                    { {"io", Traced_Shape::VarType::FLOAT3, true, CL_MEM_READ_ONLY},
                      {"id", Traced_Shape::VarType::FLOAT3, true, CL_MEM_READ_ONLY},
                      {"ivalid", Traced_Shape::VarType::FLOAT, true, CL_MEM_READ_ONLY},
                      {"indRatio", Traced_Shape::VarType::FLOAT, true, CL_MEM_READ_ONLY},
                      {"time", Traced_Shape::VarType::FLOAT, false, CL_MEM_READ_ONLY},
                      {"ro", Traced_Shape::VarType::FLOAT3, true, CL_MEM_WRITE_ONLY},
                      {"rd", Traced_Shape::VarType::FLOAT3, true, CL_MEM_WRITE_ONLY},
                      {"rvalid", Traced_Shape::VarType::FLOAT, true, CL_MEM_READ_WRITE}};


static const char* DEFAULT_RAY_TRACE =
                    "__kernel void main(__global float3* io,\n" //Incident ray origin.
                    "              __global float3* id,\n" //Incident ray direction.
                    "              __global float* ivalid,\n" //Incident ray valid.
                    "              __global float* indRatio,\n" //Refraction index ratio for each ray when hit.
                    "              __global float* time,\n" //Time constant.
                    "              __global float3* ro,\n" //Reflected/refracted ray origin.
                    "              __global float3* rd,\n" //Reflected/refracted ray direction.
                    "              __global float* rvalid) {\n" //Reflected/refracted ray valid.
                    "    uint gid = get_global_id(0);\n"
                    "    if (isequal(ivalid[gid], 0.0)) {\n"
                    "        rd[gid] = (float3)(indRatio[gid], 0, 0);\n"
                    "        ro[gid] = (float3)(0, 0, 0);\n"
                    "        rvalid[gid] = 0.0;\n"
                    "    } else {\n"
                    "        float4 cast = castRay(io[gid], id[gid], time[0]);\n" //Get ray propagation distance.
                    "        float3 pos = io[gid] + cast.x * id[gid];\n"
                    "        ro[gid] = pos;\n"
                    "        if (isequal(cast.y, -1.0)) {\n"
                    "            rvalid[gid] = 0.0;\n"
                    "            rd[gid] = id[gid];\n"
                    "        } else {\n"
                    "            rvalid[gid] = 1.0;\n"
                    "            float3 nor = calcNormal( pos, time[0]);\n"
                    "            if (isgreater(dot(nor, id[gid]),0.0)) {\n"
                    "                nor = -nor;\n"
                    "            }\n"
                    "            bool isRefract = isRefraction(id[gid], nor, 1.0/indRatio[gid]); \n"
                    "            if (isRefract) {\n"
                    "                rd[gid] = refract(id[gid], nor, 1.0/indRatio[gid]);\n"
                    "            } else {\n"
                    "                rd[gid] = reflect(id[gid], nor);\n"
                    "            }\n"
                    "        }\n"
                    "    }\n"
                    "}\n"
                    "\n";

static const std::vector<std::tuple<std::string, Traced_Shape::VarType, bool, cl_mem_flags>>
    DEFAULT_RAY_INIT_PARAMETER=
                    { {"i", Traced_Shape::VarType::FLOAT3, true, CL_MEM_READ_ONLY},
                      {"io", Traced_Shape::VarType::FLOAT3, true, CL_MEM_WRITE_ONLY},
                      {"id", Traced_Shape::VarType::FLOAT3, true, CL_MEM_WRITE_ONLY},
                      {"ic", Traced_Shape::VarType::FLOAT4, true, CL_MEM_WRITE_ONLY},
                      {"indRatio", Traced_Shape::VarType::FLOAT, true, CL_MEM_WRITE_ONLY}};

static const char* DEFAULT_RAY_INIT =
                    "__kernel void init_main(__global float3* i,\n" //The only variable to input to functions generating initial ray values.
                    "                   __global float3* io,\n" //Initial ray origin.
                    "                   __global float3* id,\n" //Initial ray direction.
                    "                   __global float3* ic,\n" //Initial ray colour.
                    "                   __global float* indRatio\n" //Initial ray index or reflection.
                    "                   ) {\n"
                    "    uint gid = get_global_id(0);\n"
                    "    io[gid] = rays_origin(i[gid]);\n"
                    "    id[gid] = rays_direction(i[gid]);\n"
                    "    ic[gid] = rays_colour(i[gid]);\n"
                    "    indRatio[gid] = rays_index(i[gid]);\n"
                    "}\n"
                    "\n";

static const char* DEFAULT_RAY_CALC_KERNEL_NAME = "main";
static const char* DEFAULT_INIT_RAY__KERNEL_NAME = "init_main";

//Required shader functions: dist, calcNormal, castRay, colour
//Required shader constant: ray_max_iter, ray_max_depth
//Ray trace -> Get normal -> Bound check -> Refraction -> Ray trace

void export_clprog_2d(const Shape_Program& shape, const Render_Opts& opts, std::ostream& out);
void export_clprog_3d(const Shape_Program& shape, const Render_Opts& opts, std::ostream& out);
void export_rays_clprog_2d(const Rays_Program& rays, const Render_Opts& opts, std::ostream& out);
void export_rays_clprog_3d(const Rays_Program& rays, const Render_Opts& opts, std::ostream& out);

void export_clprog(const Shape_Program& shape, const Render_Opts& opts, std::ostream& out)
{
    if (shape.is_2d_)
        return export_clprog_2d(shape, opts, out);
    if (shape.is_3d_)
        return export_clprog_3d(shape, opts, out);
    die("export_clprog: shape is not 2d or 3d");
}

void export_clprog(const Shape_Program& shape, const Rays_Program& rays, const Render_Opts& opts, std::ostream& out, std::ostream& initOut)
{
    export_clprog(shape, opts, out);
//    if (rays.ray_is_2d_)
//        return export_rays_clprog_2d(rays, opts, out);
    if (rays.ray_is_3d_)
        return export_rays_clprog_3d(rays, opts, initOut);
    die("export_clprog: shape is not 2d or 3d");
}

void export_rays_clprog_3d(const Rays_Program& rays, const Render_Opts& opts, std::ostream& out)
{
#ifdef OPENCL_TEST_KERNEL
    out << TEST_KERNEL;
    return;
#endif
    out <<
        DEFAULT_HEADER;

    SC_Compiler sc(out, SC_Target::glsl, rays.system());
    At_Program cx(rays);

    out << glsl_header;
    if (rays.traced_shape_) {
        // output uniform variables for parametric shape
        for (auto& p : rays.traced_shape_->param_) {
            out << "uniform " << p.second.pconfig_.sctype_ << " "
                << p.second.identifier_ << ";\n";
        }
    }
    sc.define_function("rays_origin", SC_Type::Num(3), SC_Type::Num(3),
        rays.rays_origin_fun_, cx);
    sc.define_function("rays_direction", SC_Type::Num(3), SC_Type::Num(3),
        rays.rays_direction_fun_, cx);
    sc.define_function("rays_colour", SC_Type::Num(3), SC_Type::Num(3),
        rays.rays_colour_fun_, cx);
    sc.define_function("rays_index", SC_Type::Num(3), SC_Type::Num(),
        rays.rays_index_fun_, cx);

    BBox bbox = rays.bbox_;
    if (bbox.empty2() || bbox.infinite2()) {
        out <<
        "const vec4 bbox = vec4(-10.0,-10.0,+10.0,+10.0);\n";
    } else {
        out << "const vec4 bbox = vec4("
            << bbox.xmin << ","
            << bbox.ymin << ","
            << bbox.xmax << ","
            << bbox.ymax
            << ");\n";
    }

    out <<
        DEFAULT_RAY_INIT;

}


void export_clprog_2d(const Shape_Program& shape, const Render_Opts& opts, std::ostream& out)
{
#ifdef OPENCL_TEST_KERNEL
    out << TEST_KERNEL;
    return;
#endif
    out <<
        DEFAULT_HEADER
        <<
        DEFAULT_REFLECT
        <<
        DEFAULT_REFRACTION
        <<
        DEFAULT_IS_REFRACTION;

    glsl_function_export(shape, out);

    BBox bbox = shape.bbox_;
    if (bbox.empty2() || bbox.infinite2()) {
        out <<
        "const vec4 bbox = vec4(-10.0,-10.0,+10.0,+10.0);\n";
    } else {
        out << "const vec4 bbox = vec4("
            << bbox.xmin << ","
            << bbox.ymin << ","
            << bbox.xmax << ","
            << bbox.ymax
            << ");\n";
    }

    out <<
        DEFAULT_CAST_RAY
        <<
        DEFAULT_CALC_NORMAL
        <<
        DEFAULT_RAY_TRACE;

}

void export_clprog_3d(const Shape_Program& shape, const Render_Opts& opts, std::ostream& out)
{
#ifdef OPENCL_TEST_KERNEL
    out << TEST_KERNEL;
    return;
#endif

    out <<
        DEFAULT_HEADER;

    out <<
        "const int ray_max_iter = " << opts.ray_max_iter_ << ";\n"
        "const float ray_max_depth = " << dfmt(opts.ray_max_depth_, dfmt::EXPR) << ";\n";

    out <<
        DEFAULT_REFLECT
        <<
        DEFAULT_REFRACTION
        <<
        DEFAULT_IS_REFRACTION;

    glsl_function_export(shape, out);

    BBox bbox = shape.bbox_;
    if (bbox.empty3() || bbox.infinite3()) {
        out <<
        "const vec3 bbox_min = vec3(-10.0,-10.0,-10.0);\n"
        "const vec3 bbox_max = vec3(+10.0,+10.0,+10.0);\n";
    } else {
        out
        << "const vec3 bbox_min = vec3("
            << dfmt(bbox.xmin, dfmt::EXPR) << ","
            << dfmt(bbox.ymin, dfmt::EXPR) << ","
            << dfmt(bbox.zmin, dfmt::EXPR)
            << ");\n"
        << "const vec3 bbox_max = vec3("
            << dfmt(bbox.xmax, dfmt::EXPR) << ","
            << dfmt(bbox.ymax, dfmt::EXPR) << ","
            << dfmt(bbox.zmax, dfmt::EXPR)
            << ");\n";
    }

    out <<
        DEFAULT_CAST_RAY
        <<
        DEFAULT_CALC_NORMAL
        <<
        DEFAULT_RAY_TRACE;

}


Traced_Shape::Traced_Shape(const Shape_Program& shape, const Render_Opts& opts) : Viewed_Shape(shape, opts)
{
        std::stringstream clprog;
        export_clprog(shape, opts, clprog);
        clprog_ = clprog.str();

}

Traced_Shape::Traced_Shape(const Shape_Program& shape, const Rays_Program& rays, const Render_Opts& opts) : Viewed_Shape(shape, opts)
{
        std::stringstream clprog, clinitprog;
        export_clprog(shape, rays, opts, clprog, clinitprog);
        clprog_ = clprog.str();
        clinitprog_ = clinitprog.str();

}

uint Traced_Shape::getVarIndex(const std::vector<std::tuple<std::string, Traced_Shape::VarType, bool, cl_mem_flags>>& paramSet, const std::string &varName, const VarType type, const bool isArray)
{
    for (long unsigned int i=0; i < paramSet.size(); i++) {
        auto element = paramSet[i];
        if ( varName == std::get<0>(element) &&
             type == std::get<1>(element) &&
             isArray == std::get<2>(element)) {
            return i;
        }
    }
    return -1;
}
void Traced_Shape::setInitBuffers() {
    setInitBuffers(numRays_);
}

void Traced_Shape::setInitBuffers(std::tuple<unsigned int, unsigned int, unsigned int> numRays) {
    //clear  argument data.
    argsData_.clear();

    unsigned int totalRays = std::get<0>(numRays) * std::get<1>(numRays) * std::get<2>(numRays);

    //Allocate space for data.
    cl_float3 zerofl;
    zerofl.x = 0;
    zerofl.y = 0;
    zerofl.z = 0;

    argsData_["i"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]),
                      "i", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    argsData_["io"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]),
                      "io", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    argsData_["id"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]),
                      "id", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    argsData_["indRatio"] = MemDataAttr(std::shared_ptr<cl_float[]>(new cl_float[totalRays]),
                      "indRatio", VarType::FLOAT, sizeof(cl_float) * totalRays);
    argsData_["ro"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]{zerofl}),
                      "ro", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    argsData_["rd"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]{zerofl}),
                      "rd", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    argsData_["ivalid"] = MemDataAttr(std::shared_ptr<cl_float[]>(new cl_float[totalRays]{1.0}),
                      "ivalid", VarType::FLOAT, sizeof(cl_float) * totalRays);
    argsData_["rvalid"] = MemDataAttr(std::shared_ptr<cl_float[]>(new cl_float[totalRays]{0.0}),
                      "rvalid", VarType::FLOAT, sizeof(cl_float) * totalRays);
    argsData_["time"] = MemDataAttr(std::shared_ptr<cl_float[]>(new cl_float[1]{0}),
                      "time", VarType::FLOAT, sizeof(cl_float) * 1);
    argsData_["ic"] = MemDataAttr(std::shared_ptr<cl_float3[]>(new cl_float3[totalRays]),
                      "ic", VarType::FLOAT3, sizeof(cl_float3) * totalRays);
    //Write initialRays.
    unsigned int i = 0;
    for (unsigned int a=0; a< std::get<0>(numRays);a++) {
        for (unsigned int b=0; b< std::get<1>(numRays);b++) {
            for (unsigned int c=0; c< std::get<2>(numRays);c++) {
                //Evenly spread i from 0.0 to 1.0.
                std::reinterpret_pointer_cast<cl_float3[]>(argsData_["i"].data_).get()[i].x =
                        (float)a/fmax(1.0, (float)(std::get<0>(numRays) - 1));
                std::reinterpret_pointer_cast<cl_float3[]>(argsData_["i"].data_).get()[i].y =
                        (float)b/fmax(1.0, (float)(std::get<1>(numRays) - 1));
                std::reinterpret_pointer_cast<cl_float3[]>(argsData_["i"].data_).get()[i].z =
                        (float)c/fmax(1.0, (float)(std::get<2>(numRays) - 1));
                std::reinterpret_pointer_cast<cl_float[]>(argsData_["ivalid"].data_).get()[i] = 1.0;
                std::reinterpret_pointer_cast<cl_float[]>(argsData_["rvalid"].data_).get()[i] = 0.0;
                i++;
            }
        }
    }
    numRays_ = numRays;
    finished_ = false;
}


void Traced_Shape::setInitialRays(const std::vector<Ray>& inputRays)
{

    unsigned int numRays = inputRays.size();
    setInitBuffers(std::tuple<unsigned int, unsigned int, unsigned int>(numRays, 1, 1));
    //Write initialRays.
    for (unsigned int i=0; i<numRays;i++) {
        cl_float3 o, d, c;
        cl_float r;
        o.x = inputRays[i].pos.x;
        o.y = inputRays[i].pos.y;
        o.z = inputRays[i].pos.z;
        d.x = inputRays[i].dir.x;
        d.y = inputRays[i].dir.y;
        d.z = inputRays[i].dir.z;
        r = inputRays[i].refractIndRatio;
        c.x = 1.0;
        c.y = 1.0;
        c.z = 1.0;

        std::reinterpret_pointer_cast<cl_float3[]>(argsData_["io"].data_).get()[i] = o;
        std::reinterpret_pointer_cast<cl_float3[]>(argsData_["id"].data_).get()[i] = d;
        std::reinterpret_pointer_cast<cl_float3[]>(argsData_["ic"].data_).get()[i] = c;
        std::reinterpret_pointer_cast<cl_float[]>(argsData_["indRatio"].data_).get()[i] = r;
        //std::cout << "Init rays" << std::endl;
        //std::cout << "ivalid is " << std::to_string(std::reinterpret_pointer_cast<cl_float[]>(argsData_["ivalid"].data_).get()[i]) << ", rvalid is " << std::to_string(std::reinterpret_pointer_cast<cl_float[]>(argsData_["rvalid"].data_).get()[i]) << std::endl;
    }
}

void Traced_Shape::setInitialRays() {
    rays_.clear();
    // Parse initial rays and parameters from shape.

    //(Re)create input and output memory objects. Free existing objects.
    //Set input memory objects with initial ray values.
    if (getNumRays() == 0) {
        std::vector<Ray> rays;
        rays.push_back(Ray{glm::vec3(-3.0,-1.0,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,-0.75,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,-0.5,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,-0.25,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,0,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,1.0,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,0.75,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,0.5,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        rays.push_back(Ray{glm::vec3(-3.0,0.25,0), glm::vec3(1,0,0), glm::vec4(1,1,1,10), 2.0});
        setInitialRays(rays);
    } else {
        setInitBuffers();
        calc_init_rays_ = true;
    }
    finished_ = false;
}

bool Traced_Shape::propagate() {
    bool ended = finished_ | (getNumRays() <= 0);
    float far_away = 10000;
    if (!ended) {
        //std::cout << "Propagate:" << std::endl;
        bool no_valid_rrays = true;
        for (uint i=0; i < getNumRays(); i++) {
            //Incident ray is valid?
            bool is_i_valid = (std::reinterpret_pointer_cast<cl_float[]>(argsData_["ivalid"].data_).get()[i] > 0.0);
            //Reflected ray is valid?
            bool is_r_valid = (std::reinterpret_pointer_cast<cl_float[]>(argsData_["rvalid"].data_).get()[i] > 0.0);
            no_valid_rrays &= !is_r_valid;
            //Add valid reflected rays to result.
            if (is_i_valid) {
                cl_float3 rd = std::reinterpret_pointer_cast<cl_float3[]>(argsData_["rd"].data_).get()[i];
                cl_float3 ro = std::reinterpret_pointer_cast<cl_float3[]>(argsData_["ro"].data_).get()[i];
                //cl_float3 id = std::reinterpret_pointer_cast<cl_float3[]>(argsData_["id"].data_).get()[i];
                cl_float3 io = std::reinterpret_pointer_cast<cl_float3[]>(argsData_["io"].data_).get()[i];
                rays_.push_back(Ray{glm::vec3(io.x, io.y, io.z),
                                    glm::vec3(ro.x - io.x, ro.y - io.y, ro.z - io.z),
                                    glm::vec4(1,1,1,1), 1});
                if (!is_r_valid) {
                    rays_.push_back(Ray{glm::vec3(ro.x, ro.y, ro.z),
                                    glm::vec3(rd.x * far_away,
                                              rd.y * far_away,
                                              rd.z * far_away),
                                    glm::vec4(1,0,0,1), 1});
                }
            //std::cout << "io is (" << std::to_string(io.x)
            //          << ", " << std::to_string(io.y)
            //          << ", " << std::to_string(io.z)
            //          << ")" << std::endl;
            //std::cout << "id is (" << std::to_string(id.x)
            //          << ", " << std::to_string(id.y)
            //          << ", " << std::to_string(id.z)
            //          << ")" << std::endl;
            //std::cout << "ro is (" << std::to_string(ro.x)
            //          << ", " << std::to_string(ro.y)
            //          << ", " << std::to_string(ro.z)
            //          << ")" << std::endl;
            //std::cout << "rd is (" << std::to_string(rd.x)
            //          << ", " << std::to_string(rd.y)
            //          << ", " << std::to_string(rd.z)
            //          << ")" << std::endl;
            }
            if (is_i_valid && is_r_valid) {
                //Flip reflection index for next round.
                std::reinterpret_pointer_cast<cl_float[]>(argsData_["indRatio"].data_).get()[i] =
                        1 / std::reinterpret_pointer_cast<cl_float[]>(argsData_["indRatio"].data_).get()[i];
            }
            //std::cout << "ivalid is " << std::to_string(is_i_valid) << ", rvalid is " << std::to_string(is_r_valid) << std::endl;
        }
        //check if propagation is finished, update finished_ if neccessary.
        if (no_valid_rrays) {
            finished_ = true;
            ended = true;
        } else {
            //Exchange incident ray buffers with those of reflected ray.
            argsData_["rd"].data_.swap(argsData_["id"].data_);
            argsData_["ro"].data_.swap(argsData_["io"].data_);
            argsData_["ivalid"].data_.swap(argsData_["rvalid"].data_);
        }
        //std::cout << "finished is " << std::to_string(finished_) << "ended is " << std::to_string(ended) << std::endl;
    }
    return ended;
}

std::vector<Traced_Shape::KernelParam> Traced_Shape::getKernelArgParams() {
    return getArgParams(ParamSet::KERNEL);
}

std::vector<Traced_Shape::KernelParam> Traced_Shape::getRayInitArgParams() {
    return getArgParams(ParamSet::RAY_INIT);
}

std::vector<Traced_Shape::KernelParam> Traced_Shape::getArgParams(ParamSet set) {
    std::vector<Traced_Shape::KernelParam> result;
    std::vector<std::tuple<std::string, Traced_Shape::VarType, bool, cl_mem_flags>> paramSet;

    switch (set) {
        case RAY_INIT:
            paramSet = DEFAULT_RAY_INIT_PARAMETER;
            break;
        case KERNEL:
            paramSet = DEFAULT_KERNEL_PARAMETER;
            break;
        default:
            die("getArgParams: Unknow parameter set.");
    }

    for (auto e: paramSet) {
        if (auto data = getData(std::get<0>(e))) {
            result.push_back(Traced_Shape::KernelParam(std::get<0>(e),
                    getVarIndex(paramSet, std::get<0>(e), std::get<1>(e), std::get<2>(e)),
                    std::get<1>(e), std::get<2>(e), data->size_,
                    data->data_.get(), std::get<3>(e)));
        }
    }
    return result;
}

std::optional<Traced_Shape::MemDataAttr> Traced_Shape::getData(const std::string& param) {
    return argsData_.find(param) != argsData_.end() ?
            std::optional<MemDataAttr>(argsData_[param]) : std::nullopt;
}

std::string Traced_Shape::getRayCalcKernelName() {
    return DEFAULT_RAY_CALC_KERNEL_NAME;
}

std::string Traced_Shape::getInitRayKernelName() {
    return DEFAULT_INIT_RAY__KERNEL_NAME;
}

} //namespace
