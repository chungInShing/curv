// Copyright 2021 Martin Chung
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_TRACED_GPU_PROGRAM_H
#define LIBCURV_TRACED_GPU_PROGRAM_H

#include <libcurv/gpu_program.h>
#include <libcurv/traced_shape.h>
#include <libcurv/rays.h>

namespace curv {

struct Traced_GPU_Program : GPU_Program {

    bool ray_is_2d_;
    bool ray_is_3d_;

    Traced_Shape tshape_;

    Traced_GPU_Program(Program&);

    bool recognize(Value, Render_Opts);
};

} //namespace
#endif // header guard
