// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_RAYS_H
#define LIBCURV_RAYS_H
#include <libcurv/frame.h>
#include <libcurv/sc_compiler.h>
#include <libcurv/location.h>
#include <libcurv/vec.h>
#include <libcurv/shape.h>
#include <cmath>
#include <tuple>


namespace curv {
struct Function;
struct Context;
struct System;
struct Program;
struct Phrase;
struct Render_Opts;

struct Traced_Shape;


struct Rays_Program
{
    bool ray_is_2d_;
    bool ray_is_3d_;
    BBox bbox_;
    std::tuple<unsigned int, unsigned int, unsigned int> num_rays_;

    // is_shape is initially false, becomes true after recognize() succeeds.
    bool is_shape() const { return ray_is_2d_ || ray_is_3d_; }

    System& system_;

    // describes the source code for the shape expression
    Shared<const Phrase> nub_;

    Location location() const;
    System& system() const { return system_; }
    Frame* file_frame() const { return nullptr; }
    bool recognize(Value val, Render_Opts* opts);

    // shape fields, filled in by recognize()
    Shared<Record> record_;
    Shared<const Function> rays_origin_fun_;
    Shared<const Function> rays_direction_fun_;
    Shared<const Function> rays_colour_fun_;
    Shared<const Function> rays_index_fun_;
    std::unique_ptr<Frame> rays_origin_frame_;
    std::unique_ptr<Frame> rays_direction_frame_;
    std::unique_ptr<Frame> rays_colour_frame_;
    std::unique_ptr<Frame> rays_index_frame_;

    Traced_Shape* traced_shape_ = nullptr;

    Rays_Program(Program&);
    Rays_Program(const Rays_Program& rays, Shared<Record> r, Traced_Shape* vs);
    Rays_Program(System& sys, Shared<const Phrase> nub)
        : system_(sys), nub_(std::move(nub))
    {}

};

} //namespace
#endif
