// Copyright 2016-2020 Doug Moen
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include <libcurv/gpu_program.h>

#include <libcurv/context.h>
#include <libcurv/exception.h>
#include <libcurv/format.h>
#include <libcurv/program.h>

namespace curv {

GPU_Program::GPU_Program(Program& prog)
:
    system_(prog.system()),
    nub_(nub_phrase(prog.phrase_))
{
    // mark initial state (no shape has been recognized yet)
    is_2d_ = false;
    is_3d_ = false;
}

Location
GPU_Program::location() const
{
    return nub_->location();
}

bool
GPU_Program::recognize(Value val, Render_Opts opts)
{
    if (location().source().type_ == Source::Type::gpu) {
        // Note: throw exception if val is not a GPU program.
        static Symbol_Ref is_2d_key = make_symbol("is_2d");
        static Symbol_Ref is_3d_key = make_symbol("is_3d");
        static Symbol_Ref bbox_key = make_symbol("bbox");
        static Symbol_Ref shader_key = make_symbol("shader");
        static Symbol_Ref parameters_key = make_symbol("parameters");
        static Symbol_Ref name_key = make_symbol("name");
        static Symbol_Ref value_key = make_symbol("value");
        static Symbol_Ref label_key = make_symbol("label");
        static Symbol_Ref config_key = make_symbol("config");

        At_Program cx(*this);
        auto r = val.to<Record>(cx);

        is_2d_ = r->getfield(is_2d_key, cx).to_bool(At_Field("is_2d", cx));
        is_3d_ = r->getfield(is_3d_key, cx).to_bool(At_Field("is_3d", cx));
        if (!is_2d_ && !is_3d_)
            throw Exception(cx,
                "at least one of is_2d and is_3d must be true");

        bbox_ = BBox::from_value(
            r->getfield(bbox_key, cx),
            At_Field("bbox", cx));

        vshape_.frag_ =
            value_to_string(r->getfield(shader_key,cx),
                Fail::hard, At_Field("shader",cx))
            ->c_str();

        At_Field pcx("parameters",cx);
        auto parameters = r->getfield(parameters_key, cx).to<List>(pcx);
        At_Index picx(0, pcx);
        for (auto p : *parameters) {
            auto prec = p.to<Record>(picx);
            auto name =
                value_to_string(prec->getfield(name_key, picx),
                    Fail::hard, At_Field("name",picx))
                ->c_str();
            auto label =
                value_to_string(prec->getfield(label_key, picx),
                    Fail::hard, At_Field("label",picx))
                ->c_str();
            Picker::Config config(prec->getfield(config_key, picx),
                At_Field("config", picx));
            auto state_val = prec->getfield(value_key, picx);
            Picker::State state(config.type_, state_val, At_Field("value",picx));
            vshape_.param_.insert(
                std::pair<const std::string,Viewed_Shape::Parameter>{
                    label,
                    Viewed_Shape::Parameter{name, config, state}});
            ++picx.index_;
        }

        return true;
    }
    Shape_Program shape(system_, nub_);
    if (!shape.recognize(val, &opts))
        return false;
    is_2d_ = shape.is_2d_;
    is_3d_ = shape.is_3d_;
    bbox_ =  shape.bbox_;
    Viewed_Shape vshape(shape, opts);
    std::swap(vshape_, vshape);
    return true;
}

void
GPU_Program::write_json(std::ostream& out) const
{
    out << "{"
        << "\"is_2d\":" << (is_2d_ ? "true" : "false")
        << ",\"is_3d\":" << (is_3d_ ? "true" : "false")
        << ",\"bbox\":[[" << dfmt(bbox_.xmin, dfmt::JSON)
            << "," << dfmt(bbox_.ymin, dfmt::JSON)
            << "," << dfmt(bbox_.zmin, dfmt::JSON)
            << "],[" << dfmt(bbox_.xmax, dfmt::JSON)
            << "," << dfmt(bbox_.ymax, dfmt::JSON)
            << "," << dfmt(bbox_.zmax, dfmt::JSON)
        << "]],";
    vshape_.write_json(out);
    out << "}";
}

void
GPU_Program::write_curv(std::ostream& out) const
{
    out << "{\n"
        << "  is_2d: " << Value{is_2d_} << ";\n"
        << "  is_3d: " << Value{is_3d_} << ";\n"
        << "  bbox: [[" << dfmt(bbox_.xmin, dfmt::JSON)
            << "," << dfmt(bbox_.ymin, dfmt::JSON)
            << "," << dfmt(bbox_.zmin, dfmt::JSON)
            << "],[" << dfmt(bbox_.xmax, dfmt::JSON)
            << "," << dfmt(bbox_.ymax, dfmt::JSON)
            << "," << dfmt(bbox_.zmax, dfmt::JSON)
        << "]];\n";
    vshape_.write_curv(out);
    out << "}\n";
}

} // namespace
