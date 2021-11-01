// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include "libcurv/symbol.h"
#include <libcurv/program.h>
#include <libcurv/context.h>
#include <libcurv/exception.h>
#include <libcurv/rays.h>

namespace curv {
Rays_Program::Rays_Program(
    Program& prog)
:
    system_(prog.system()),
    nub_(nub_phrase(prog.phrase_))
{
    // mark initial state (no shape has been recognized yet)
    ray_is_2d_ = false;
    ray_is_3d_ = false;
}

Location Rays_Program::location() const
{
    return nub_->location();
}

Rays_Program::Rays_Program(
    const Rays_Program& rays,
    Shared<Record> r,
    Traced_Shape* vs)
:
    system_(rays.system_),
    nub_(rays.nub_),
    record_(r),
    traced_shape_(vs)
{
    static Symbol_Ref rays_origin_key = make_symbol("rays_origin");
    static Symbol_Ref rays_direction_key = make_symbol("rays_direction");
    static Symbol_Ref rays_colour_key = make_symbol("rays_colour");
    static Symbol_Ref rays_index_key = make_symbol("rays_index");
    static Symbol_Ref num_rays_key = make_symbol("nrays");

    ray_is_2d_ = rays.ray_is_2d_;
    ray_is_3d_ = rays.ray_is_3d_;
    bbox_ = rays.bbox_;

    At_Program cx(*this);

    if (r->hasfield(rays_origin_key))
        rays_origin_fun_ = value_to_function(r->getfield(rays_origin_key, cx), cx);
    else
        throw Exception{cx, stringify(
            "bad parametric shape: call result has no 'rays_origin' field: ", r)};
    if (r->hasfield(rays_direction_key))
        rays_direction_fun_ = value_to_function(r->getfield(rays_direction_key, cx), cx);
    else
        throw Exception{cx, stringify(
            "bad parametric shape: call result has no 'rays_direction' field: ", r)};
    if (r->hasfield(rays_colour_key))
        rays_colour_fun_ = value_to_function(r->getfield(rays_colour_key, cx), cx);
    else
        throw Exception{cx, stringify(
            "bad parametric shape: call result has no 'rays_colour' field: ", r)};
    if (r->hasfield(rays_index_key))
        rays_index_fun_ = value_to_function(r->getfield(rays_index_key, cx), cx);
    else
        throw Exception{cx, stringify(
            "bad parametric shape: call result has no 'rays_index' field: ", r)};
    if (r->hasfield(num_rays_key))
        rays_index_fun_ = value_to_function(r->getfield(num_rays_key, cx), cx);
    else
        throw Exception{cx, stringify(
            "bad parametric shape: call result has no 'nrays' field: ", r)};
}

bool Rays_Program::recognize(Value val, Render_Opts* opts)
{
    static Symbol_Ref ray_is_2d_key = make_symbol("ray_is_2d");
    static Symbol_Ref ray_is_3d_key = make_symbol("ray_is_3d");
    static Symbol_Ref rays_origin_key = make_symbol("rays_origin");
    static Symbol_Ref rays_direction_key = make_symbol("rays_direction");
    static Symbol_Ref rays_colour_key = make_symbol("rays_colour");
    static Symbol_Ref rays_index_key = make_symbol("rays_index");
    static Symbol_Ref num_rays_key = make_symbol("nrays");

    At_Program cx(*this);
    auto r = val.maybe<Record>();

    if (r == nullptr) return false;
    Value ray_is_2d_val = r->find_field(ray_is_2d_key, cx);
    if (ray_is_2d_val.is_missing()) return false;
    Value ray_is_3d_val = r->find_field(ray_is_3d_key, cx);
    if (ray_is_3d_val.is_missing()) return false;
    Value rays_origin_val = r->find_field(rays_origin_key, cx);
    if (rays_origin_val.is_missing()) return false;
    Value rays_direction_val = r->find_field(rays_direction_key, cx);
    if (rays_direction_val.is_missing()) return false;
    Value rays_colour_val = r->find_field(rays_colour_key, cx);
    if (rays_colour_val.is_missing()) return false;
    Value rays_index_val = r->find_field(rays_index_key, cx);
    if (rays_index_val.is_missing()) return false;
    Value num_rays_val = r->find_field(num_rays_key, cx);
    if (num_rays_val.is_missing()) return false;

    ray_is_2d_ = ray_is_2d_val.to_bool(At_Field("ray_is_2d", cx));
    ray_is_3d_ = ray_is_3d_val.to_bool(At_Field("ray_is_3d", cx));
    auto ray_list = num_rays_val.to<List>(cx);
    switch (ray_list->size()) {
        case 1:
            std::get<0>(num_rays_) = ray_list->val_at(0).to_int(0, std::numeric_limits<int>::max(), cx);
            std::get<1>(num_rays_) = 1;
            std::get<2>(num_rays_) = 1;
            break;
        case 2:
            std::get<0>(num_rays_) = ray_list->val_at(0).to_int(0, std::numeric_limits<int>::max(), cx);
            std::get<1>(num_rays_) = ray_list->val_at(1).to_int(0, std::numeric_limits<int>::max(), cx);
            std::get<2>(num_rays_) = 1;
            break;
        case 3:
            std::get<0>(num_rays_) = ray_list->val_at(0).to_int(0, std::numeric_limits<int>::max(), cx);
            std::get<1>(num_rays_) = ray_list->val_at(1).to_int(0, std::numeric_limits<int>::max(), cx);
            std::get<2>(num_rays_) = ray_list->val_at(2).to_int(0, std::numeric_limits<int>::max(), cx);
            break;
        default:
            throw Exception(cx, "nrays must be a list with 1 to 3 elements");

    }

    if (!ray_is_2d_ && !ray_is_3d_)
        throw Exception(cx,
            "at least one of ray_is_2d and ray_is_3d must be true");

    rays_origin_fun_ = value_to_function(rays_origin_val, At_Field("rays_origin", cx));
    rays_origin_frame_ = Frame::make(
        rays_origin_fun_->nslots_, system_, nullptr, nullptr, nullptr);
    rays_direction_fun_ = value_to_function(rays_direction_val, At_Field("rays_direction", cx));
    rays_direction_frame_ = Frame::make(
        rays_direction_fun_->nslots_, system_, nullptr, nullptr, nullptr);
    rays_colour_fun_ = value_to_function(rays_colour_val, At_Field("rays_colour", cx));
    rays_colour_frame_ = Frame::make(
        rays_colour_fun_->nslots_, system_, nullptr, nullptr, nullptr);
    rays_index_fun_ = value_to_function(rays_index_val, At_Field("rays_index", cx));
    rays_index_frame_ = Frame::make(
        rays_index_fun_->nslots_, system_, nullptr, nullptr, nullptr);
    record_ = r;

    return true;
}

} //namespace
