// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include <libcurv/program.h>
#include <libcurv/context.h>
#include <libcurv/format.h>
#include <libcurv/exception.h>
#include <libcurv/rays.h>
#include <libcurv/traced_gpu_program.h>

namespace curv {
Traced_GPU_Program::Traced_GPU_Program(Program& prog)
:
    GPU_Program(prog)
{
    ray_is_2d_ = false;
    ray_is_3d_ = false;

}

bool
Traced_GPU_Program::recognize(Value val, Render_Opts opts)
{
    bool result = GPU_Program::recognize(val, opts);
    if (result) {
        At_Program cx(*this);
        auto r = val.to<Record>(cx);
        static Symbol_Ref ray_is_2d_key = make_symbol("ray_is_2d");
        static Symbol_Ref ray_is_3d_key = make_symbol("ray_is_3d");

        ray_is_2d_ = r->getfield(ray_is_2d_key, cx).to_bool(At_Field("ray_is_2d", cx));
        ray_is_3d_ = r->getfield(ray_is_3d_key, cx).to_bool(At_Field("ray_is_3d", cx));
        if (!ray_is_2d_ && !ray_is_3d_)
            throw Exception(cx,
                "at least one of ray_is_2d and ray_is_3d must be true");

        //static Symbol_Ref parameters_key = make_symbol("parameters");
        //static Symbol_Ref name_key = make_symbol("name");
        //static Symbol_Ref value_key = make_symbol("value");
        //static Symbol_Ref label_key = make_symbol("label");
        //static Symbol_Ref config_key = make_symbol("config");
        //At_Field pcx("parameters",cx);
        //auto parameters = r->getfield(parameters_key, cx).to<List>(pcx);
        //At_Index picx(0, pcx);
        //for (auto p : *parameters) {
        //    auto prec = p.to<Record>(picx);
        //    auto name =
        //        value_to_string(prec->getfield(name_key, picx),
        //            Fail::hard, At_Field("name",picx))
        //        ->c_str();
        //    auto label =
        //        value_to_string(prec->getfield(label_key, picx),
        //            Fail::hard, At_Field("label",picx))
        //        ->c_str();
        //    Picker::Config config(prec->getfield(config_key, picx),
        //        At_Field("config", picx));
        //    auto state_val = prec->getfield(value_key, picx);
        //    Picker::State state(config.type_, state_val, At_Field("value",picx));
        //    tshape_.param_.insert(
        //        std::pair<const std::string,Traced_Shape::Parameter>{
        //            label,
        //            Traced_Shape::Parameter{name, config, state}});
        //    ++picx.index_;
        //}

        Shape_Program shape(system_, nub_);
        if (!shape.recognize(val, &opts))
            return false;
        Rays_Program rays(system_, nub_);
        if (!rays.recognize(val, &opts))
            return false;
        ray_is_2d_ = rays.ray_is_2d_;
        ray_is_3d_ = rays.ray_is_3d_;

        Traced_Shape tshape(shape, rays, opts);
        std::swap(tshape_, tshape);
    }
    return result;
}

} //namespace
