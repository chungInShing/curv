// Copyright 2016-2020 Doug Moen
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#include <cctype>
#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>

#include <libcurv/bool.h>
#include <libcurv/context.h>
#include <libcurv/die.h>
#include <libcurv/dtostr.h>
#include <libcurv/exception.h>
#include <libcurv/function.h>
#include <libcurv/num.h>
#include <libcurv/meaning.h>
#include <libcurv/optimizer.h>
#include <libcurv/picker.h>
#include <libcurv/prim_expr.h>
#include <libcurv/reactive.h>
#include <libcurv/sc_compiler.h>
#include <libcurv/sc_context.h>
#include <libcurv/system.h>

namespace curv {

struct SC_Data_Ref : public Operation
{
    SC_Value val_;
    SC_Data_Ref(Shared<const Phrase> src, SC_Value v)
    : Operation(std::move(src)), val_(v)
    {}
    SC_Value sc_eval(SC_Frame&) const override { return val_; }
    void exec(Frame&, Executor&) const override { }
};

// This is the main entry point into the Shape Compiler.
void
SC_Compiler::define_function(
    const char* name, SC_Type param_type, SC_Type result_type,
    Shared<const Function> func, const Context& cx)
{
    define_function(
        name,
        std::vector<SC_Type>{param_type},
        result_type,
        func,
        cx);
}

void
SC_Compiler::define_function(
    const char* name,
    std::vector<SC_Type> param_types,
    SC_Type result_type,
    Shared<const Function> func,
    const Context& cx)
{
    begin_function();

    // function prologue
    if (target_ == SC_Target::cpp)
        out_ << "extern \"C\" void " << name << "(";
    else
        out_ << result_type << " " << name << "(";
    bool first = true;
    std::vector<SC_Value> params;
    int n = 0;
    for (auto& ty : param_types) {
        params.push_back(newvalue(ty));
        if (!first) out_ << ", ";
        first = false;
        if (target_ == SC_Target::cpp)
            out_ << "const " << ty << "* param" << n++;
        else
            out_ << ty << " " << params.back();
    }
    if (target_ == SC_Target::cpp) {
        if (!first) out_ << ", ";
        out_ << result_type << "* result)\n";
    } else
        out_ << ")\n";
    out_ << "{\n";
    if (target_ == SC_Target::cpp) {
        n = 0;
        for (unsigned i = 0; i < params.size(); ++i) {
            out_ << "  " << param_types[i] << " " << params[i]
                 << " = *param" << n++ << ";\n";
        }
    }

    // function body
    auto f = SC_Frame::make(0, *this, &cx, nullptr, nullptr);
    Shared<Operation> arg_expr;
    if (params.size() == 1)
        arg_expr = make<SC_Data_Ref>(nullptr, params[0]);
    else {
        auto param_list = List_Expr::make(params.size(),nullptr);
        for (unsigned i = 0; i < params.size(); ++i) {
            param_list->at(i) = make<SC_Data_Ref>(nullptr, params[i]);
        }
        arg_expr = std::move(param_list);
    }
    auto result = func->sc_call_expr(*arg_expr, nullptr, *f);
    if (result.type != result_type) {
        throw Exception(cx, stringify(name," function returns ",result.type));
    }
    end_function();

    // function epilogue
    if (target_ == SC_Target::cpp) {
        out_ << "  *result = " << result << ";\n";
    } else {
        out_ << "  return " << result << ";\n";
    }
    out_ << "}\n";
}

void
SC_Compiler::begin_function()
{
    valcount_ = 0;
    valcache_.clear();
    opcaches_.clear();
    opcaches_.emplace_back(Op_Cache{});
    constants_.str("");
    body_.str("");
}

void
SC_Compiler::end_function()
{
    out_ << "  /* constants */\n";
    out_ << constants_.str();
    out_ << "  /* body */\n";
    out_ << body_.str();
}

struct Set_Purity
{
    SC_Compiler& sc_;
    bool previous_purity_;
    Set_Purity(SC_Compiler& sc, bool purity)
    :
        sc_(sc),
        previous_purity_(sc.in_constants_)
    {
        sc_.in_constants_ = purity;
    }
    ~Set_Purity()
    {
        sc_.in_constants_ = previous_purity_;
    }
};

// Wrapper for Operation::sc_eval(f), does common subexpression elimination.
SC_Value sc_eval_op(SC_Frame& f, const Operation& op)
{
#if OPTIMIZE
    if (!op.pure_) {
        Set_Purity pu(f.sc_, false);
        return op.sc_eval(f);
    }
    // 'op' is a uniform expression, consisting of pure operations at interior
    // nodes and Constants at leaf nodes. There can be no variable references
    // (eg, no Local_Data_Ref ops), other than uniform variables in reactive values.
    // What follows is a limited form of common subexpression elimination
    // which reduces code size when reactive values are used.
    for (Op_Cache& opcache : f.sc_.opcaches_) {
        auto cached = opcache.find(share(op));
        if (cached != opcache.end())
            return cached->second;
    }
    Set_Purity pu(f.sc_, true);
    auto val = op.sc_eval(f);
    f.sc_.opcaches_.back()[share(op)] = val;
    return val;
#else
    return op.sc_eval(f);
#endif
}

SC_Value sc_eval_expr(SC_Frame& f, const Operation& op, SC_Type type)
{
    SC_Value arg = sc_eval_op(f, op);
    if (arg.type != type) {
        throw Exception(At_SC_Phrase(op.syntax_, f),
            stringify("wrong argument type: expected ",type,", got ",arg.type));
    }
    return arg;
}

void
sc_put_list(
    const List& list, SC_Type ty,
    const At_SC_Phrase& cx, std::ostream& out);

// Write a value to 'out' as a GLSL/C++ initializer expression.
// As a side effect, write GLSL code to f.out when evaluating reactive values.
// At present, reactive values can occur anywhere in an array initializer.
void
sc_put_value(Value val, SC_Type ty, const At_SC_Phrase& cx, std::ostream& out)
{
    if (auto re = val.maybe<Reactive_Expression>()) {
        auto f2 = SC_Frame::make(0, cx.call_frame_.sc_, nullptr,
            &cx.call_frame_, &*cx.phrase_);
        auto result = sc_eval_op(*f2, *re->expr_);
        out << result;
    }
    else if (auto uv = val.maybe<Uniform_Variable>()) {
        out << uv->identifier_;
    }
    else if (ty.is_num()) {
        double num = val.to_num(cx);
        out << dfmt(num, dfmt::EXPR);
    }
    else if (ty.is_bool()) {
        bool b = val.to_bool(cx);
        out << (b ? "true" : "false");
    }
    else if (ty.is_bool32()) {
        Shared<const List> bl = val.to<const List>(cx);
        unsigned bn = bool32_to_nat(bl, cx);
        out << bn << "u";
    }
    else if (ty.is_vec() || ty.is_mat()) {
        Shared<const List> list = val.to<const List>(cx);
        list->assert_size(ty.count(), cx);
        out << ty << "(";
        sc_put_list(*list, ty.elem_type(), cx, out);
        out << ")";
    }
    else if (ty.plex_array_rank() > 0) {
        auto list = val.to<List>(cx);
        list->assert_size(ty.plex_array_dim(0), cx);
        sc_put_list(*list, ty.elem_type(), cx, out);
    }
    else {
        throw Exception(cx, stringify(
            "internal error at sc_put_value: ", val, ": ", ty));
    }
}

void
sc_put_list(
    const List& list, SC_Type ety,
    const At_SC_Phrase& cx, std::ostream& out)
{
    bool first = true;
    for (auto e : list) {
        if (!first) out << ",";
        first = false;
        sc_put_value(e, ety, cx, out);
    }
}

SC_Value sc_eval_const(SC_Frame& f, Value val, const Phrase& syntax)
{
#if OPTIMIZE
    auto cached = f.sc_.valcache_.find(val);
    if (cached != f.sc_.valcache_.end())
        return cached->second;
#endif
    Set_Purity pu(f.sc_, true);
    At_SC_Phrase cx(share(syntax), f);

    auto ty = sc_type_of(val);
    if (ty.is_error()) {
        throw Exception(At_SC_Phrase(share(syntax), f),
            stringify("value ",val," is not supported "));
    }

    String_Builder init;
    sc_put_value(val, ty, cx, init);
    auto initstr = init.get_string();
    SC_Value result = f.sc_.newvalue(ty);
    if (ty.is_plex()) {
        f.sc_.out()
            << "  " << ty << " " << result << " = "
            << *initstr << ";\n";
    } else {
        SC_Type ety = ty.plex_array_base();
        if (f.sc_.target_ == SC_Target::cpp) {
            f.sc_.out() << "  " << ety << " " << result << "[] = {"
                << *initstr << "};\n";
        } else {
            f.sc_.out() << "  " << ty << " " << result << " = " << ty << "("
                << *initstr << ");\n";
        }
    }

    f.sc_.valcache_[val] = result;
    return result;
}

SC_Value Operation::sc_eval(SC_Frame& f) const
{
    throw Exception(At_SC_Phrase(syntax_, f), stringify(
        "this expression is not supported: ",
        boost::core::demangle(typeid(*this).name())));
}

void Operation::sc_exec(SC_Frame& f) const
{
    throw Exception(At_SC_Phrase(syntax_, f), stringify(
        "this action is not supported: ",
        boost::core::demangle(typeid(*this).name())));
}

SC_Value Constant::sc_eval(SC_Frame& f) const
{
    return sc_eval_const(f, value_, *syntax_);
}

bool sc_try_extend(SC_Frame& f, SC_Value& a, SC_Type rtype);

// val is a scalar. rtype is an array type: could be a vec or a matrix.
// Convert 'val' to type 'rtype' by replicating the value across the elements
// of an array (aka broadcasting). If this can be done, then update variable
// 'val' in place with a new value of type 'rtype' and return true.
bool sc_try_broadcast(SC_Frame& f, SC_Value& val, SC_Type rtype)
{
    if (!sc_try_extend(f, val, rtype.elem_type())) return false;
    SC_Value result = f.sc_.newvalue(rtype);
    f.sc_.out() << "  "<<rtype<<" "<<result<<" = "<<rtype<<"(";
    if (rtype.is_bool32()) {
        f.sc_.out() << "-int("<<val<<")";
    } else if (rtype.is_vec()) {
        f.sc_.out() << val;
    } else if (rtype.is_mat()) {
        unsigned n = rtype.count();
        for (unsigned i = 0; i < n; ++i) {
            if (i > 0) f.sc_.out() << ",";
            f.sc_.out() << val;
        }
    } else
        die("sc_try_broadcast: unsupported list type");
    f.sc_.out() << ");\n";
    val = result;
    return true;
}

// 'a' is a list, 'rtype' is a list type, both have the same count.
bool sc_try_elementwise(SC_Frame& f, SC_Value& a, SC_Type rtype)
{
    unsigned count = rtype.count();
    SC_Type etype = rtype.elem_type();
    SC_Value elem[SC_Type::MAX_MAT_COUNT];
    for (unsigned i = 0; i < count; ++i) {
        elem[i] = sc_vec_element(f, a, i);
        if (!sc_try_extend(f, elem[i], etype))
            return false;
    }
    SC_Value result = f.sc_.newvalue(rtype);
    f.sc_.out() << "  "<<rtype<<" "<<result<<" = "<<rtype<<"(";
    for (unsigned i = 0; i < count; ++i) {
        if (i > 0) f.sc_.out() << ",";
        f.sc_.out() << elem[i];
    }
    f.sc_.out() << ");\n";
    a = result;
    return true;
}

// 'val' is a scalar or array. 'rtype' is a type with a rank >= rank of 'val'.
// Try to extend the value 'val' to have type 'rtype' using broadcasting and
// elementwise extension. If this is successful (the types are compatible),
// then update the variable 'val' with the new value and return true.
bool sc_try_extend(SC_Frame& f, SC_Value& a, SC_Type rtype)
{
    if (a.type == rtype) return true;
    if (a.type.is_list() && rtype.is_list()) {
        if (a.type.count() != rtype.count()) return false;
        return sc_try_elementwise(f, a, rtype);
    }
    if (rtype.is_list())
        return sc_try_broadcast(f, a, rtype);
    return false;
}

bool sc_try_unify(SC_Frame& f, SC_Value& a, SC_Value& b)
{
    if (a.type == b.type) return true;
    if (a.type.is_list() && b.type.is_list()) {
        if (a.type.count() != b.type.count()) return false;
        if (a.type.rank() < b.type.rank())
            return sc_try_elementwise(f, a, b.type);
        if (a.type.rank() > b.type.rank())
            return sc_try_elementwise(f, b, a.type);
    }
    else if (a.type.is_list())
        return sc_try_broadcast(f, b, a.type);
    else if (b.type.is_list())
        return sc_try_broadcast(f, a, b.type);
    return false;
}

// Error if a or b is not a plex.
// Succeed if a and b have the same (plex) type, or they can be converted
// to a common type using broadcasting and elementwise extension.
void sc_plex_unify(SC_Frame& f, SC_Value& a, SC_Value& b, const Context& cx)
{
    if (!a.type.is_plex())
        throw Exception(cx,
            stringify("argument with type ",a.type," is not a Plex"));
    if (!b.type.is_plex())
        throw Exception(cx,
            stringify("argument with type ",b.type," is not a Plex"));
    if (sc_try_unify(f, a, b))
        return;
    throw Exception(cx, stringify(
        "Can't convert ",a.type," and ",b.type," to a common type"));
}

// Evaluate an expression to a constant at SC compile time,
// or abort if it isn't a constant.
Value sc_constify(const Operation& op, SC_Frame& f)
{
    if (auto c = dynamic_cast<const Constant*>(&op))
        return c->value_;
    else if (auto dot = dynamic_cast<const Dot_Expr*>(&op)) {
        Value base = sc_constify(*dot->base_, f);
        if (dot->selector_.id_ != nullptr)
            return base.at(dot->selector_.id_->symbol_,
                At_SC_Phrase(op.syntax_, f));
        else
            throw Exception(At_SC_Phrase(dot->selector_.expr_->syntax_, f),
                "not an identifier");
    }
    else if (auto ref = dynamic_cast<const Nonlocal_Data_Ref*>(&op))
        return f.nonlocals_->at(ref->slot_);
    else if (auto ref = dynamic_cast<const Symbolic_Ref*>(&op)) {
        auto b = f.nonlocals_->dictionary_->find(ref->name_);
        assert(b != f.nonlocals_->dictionary_->end());
        return f.nonlocals_->get(b->second);
    }
    else if (auto list = dynamic_cast<const List_Expr*>(&op)) {
        Shared<List> listval = List::make(list->size());
        for (size_t i = 0; i < list->size(); ++i) {
            (*listval)[i] = sc_constify(*(*list)[i], f);
        }
        return {listval};
    } else if (auto neg = dynamic_cast<const Negative_Expr*>(&op)) {
        Value arg = sc_constify(*neg->arg_, f);
        if (arg.is_num())
            return Value(-arg.to_num_unsafe());
    }
    throw Exception(At_SC_Phrase(op.syntax_, f),
        "not a constant");
}

bool sc_try_constify(Operation& op, SC_Frame& f, Value& val)
{
    try {
        val = sc_constify(op, f);
        return true;
    } catch (Exception&) {
        return false;
    }
}

bool sc_try_eval(Operation& op, SC_Frame& f, SC_Value& val)
{
    try {
        val = sc_eval_op(f, op);
        return true;
    } catch (Exception&) {
        return false;
    }
}

SC_Value Block_Op::sc_eval(SC_Frame& f) const
{
    statements_.sc_exec(f);
    return sc_eval_op(f, *body_);
}
void Block_Op::sc_exec(SC_Frame& f) const
{
    statements_.sc_exec(f);
    body_->sc_exec(f);
}

SC_Value Do_Expr::sc_eval(SC_Frame& f) const
{
    actions_->sc_exec(f);
    return sc_eval_op(f, *body_);
}

void Compound_Op_Base::sc_exec(SC_Frame& f) const
{
    for (auto s : *this)
        s->sc_exec(f);
}

void Scope_Executable::sc_exec(SC_Frame& f) const
{
    for (auto action : actions_)
        action->sc_exec(f);
}
void Null_Action::sc_exec(SC_Frame&) const
{
}

void
Local_Locative::sc_print(SC_Frame& f) const
{
    f.sc_.out() << f[slot_];
}

void
Indexed_Locative::sc_print(SC_Frame& f) const
{
    // TODO: ensure that base_ is a vector
    base_->sc_print(f);
    int i = 0;
    // convert index_ to i
    auto list = cast<List_Expr>(index_);
    if (list == nullptr || list->size() != 1)
        throw Exception(At_SC_Phrase(index_->syntax_, f),
            "expected '[index]' expression");
    // i = sc_eval_index_expr() might work if we had an SC_Value for base_
    // TODO: restrict range of i based on size of vector
    Value ival = sc_constify(*list->at(0), f);
    i = ival.to_int(0, 3, At_SC_Phrase(index_->syntax_, f));
    f.sc_.out() << '[' << i << ']';
}
void
Lens_Locative::sc_print(SC_Frame& f) const
{
    // TODO: ensure that base_ is a vector
    base_->sc_print(f);
    // i = sc_eval_index_expr() might work if we had an SC_Value for base_
    // TODO: restrict range of i based on size of vector
    Value ival = sc_constify(*lens_, f);
    int i = ival.to_int(0, 3, At_SC_Phrase(lens_->syntax_, f));
    f.sc_.out() << '[' << i << ']';
}

void
Locative::sc_print(SC_Frame& f) const
{
    throw Exception(At_SC_Phrase(syntax_, f), "expression is not assignable");
}

void
Assignment_Action::sc_exec(SC_Frame& f) const
{
    SC_Value val = sc_eval_op(f, *expr_);
    f.sc_.out() << "  ";
    locative_->sc_print(f);
    f.sc_.out() << "="<<val<<";\n";
}
void
Data_Setter::sc_exec(SC_Frame& f) const
{
    assert(module_slot_ == (slot_t)(-1));
    pattern_->sc_exec(*definiens_, f, f);
}

char gl_index_letter(Value k, unsigned vecsize, const Context& cx)
{
    auto num = k.to_num_or_nan();
    if (num == 0.0)
        return 'x';
    if (num == 1.0)
        return 'y';
    if (num == 2.0 && vecsize > 2)
        return 'z';
    if (num == 3.0 && vecsize > 3)
        return 'w';
    throw Exception(cx,
        stringify("got ",k,", expected 0..",vecsize-1));
}

// compile array[i] expression
SC_Value sc_eval_index_expr(SC_Value array, Operation& index, SC_Frame& f)
{
    Value k;
    if (array.type.is_vec() && sc_try_constify(index, f, k)) {
        // A vector with a constant index. Swizzling is supported.
        if (auto list = k.maybe<List>()) {
            if (list->size() < 2 || list->size() > 4) {
                throw Exception(At_SC_Phrase(index.syntax_, f),
                    "list index vector must have between 2 and 4 elements");
            }
            char swizzle[5];
            memset(swizzle, 0, 5);
            for (size_t i = 0; i <list->size(); ++i) {
                swizzle[i] = gl_index_letter((*list)[i],
                    array.type.count(),
                    At_Index(i, At_SC_Phrase(index.syntax_, f)));
            }
            SC_Value result =
                f.sc_.newvalue(
                    SC_Type::Vec(array.type.elem_type(), list->size()));
            f.sc_.out() << "  " << result.type << " "<< result<<" = ";
            if (f.sc_.target_ == SC_Target::glsl) {
                // use GLSL swizzle syntax: v.xyz
                f.sc_.out() <<array<<"."<<swizzle;
            } else {
                // fall back to a vector constructor: vec3(v.x,v.y,v.z)
                f.sc_.out() << result.type << "(";
                bool first = true;
                for (size_t i = 0; i < list->size(); ++i) {
                    if (!first)
                        f.sc_.out() << ",";
                    first = false;
                    f.sc_.out() << array << "." << swizzle[i];
                }
                f.sc_.out() << ")";
            }
            f.sc_.out() <<";\n";
            return result;
        }
        const char* arg2 = nullptr;
        auto num = k.to_num_or_nan();
        if (num == 0.0)
            arg2 = ".x";
        else if (num == 1.0)
            arg2 = ".y";
        else if (num == 2.0 && array.type.count() > 2)
            arg2 = ".z";
        else if (num == 3.0 && array.type.count() > 3)
            arg2 = ".w";
        if (arg2 == nullptr)
            throw Exception(At_SC_Phrase(index.syntax_, f),
                stringify("got ",k,", expected 0..",
                    array.type.count()-1));

        SC_Value result = f.sc_.newvalue(array.type.elem_type());
        f.sc_.out() << "  " << result.type << " " << result << " = "
            << array << arg2 <<";\n";
        return result;
    }
    // An array of numbers, indexed with a number.
    if (array.type.plex_array_rank() > 1) {
        throw Exception(At_SC_Phrase(index.syntax_,f), stringify(
            "can't index a ", array.type.plex_array_rank(), "D array of ",
            array.type.plex_array_base(), " with a single index"));
    }
    auto ix = sc_eval_expr(f, index, SC_Type::Num());
    SC_Value result = f.sc_.newvalue(array.type.elem_type());
    f.sc_.out() << "  " << result.type << " " << result << " = "
             << array << "[int(" << ix << ")];\n";
    return result;
}

// compile array[i,j] expression
SC_Value sc_eval_index2_expr(
    SC_Value array, Operation& op_ix1, Operation& op_ix2, SC_Frame& f,
    const Context& acx)
{
    auto ix1 = sc_eval_expr(f, op_ix1, SC_Type::Num());
    auto ix2 = sc_eval_expr(f, op_ix2, SC_Type::Num());
    switch (array.type.plex_array_rank()) {
    case 2:
      {
        // 2D array of number or vector. Not supported by GLSL 1.5,
        // so we emulate this type using a 1D array.
        // Index value must be [i,j], can't use a single index.
        SC_Value result = f.sc_.newvalue(array.type.plex_array_base());
        f.sc_.out() << "  " << result.type << " " << result << " = " << array
                 << "[int(" << ix1 << ")*" << array.type.plex_array_dim(1)
                 << "+" << "int(" << ix2 << ")];\n";
        return result;
      }
    case 1:
        if (array.type.plex_array_base().rank() == 1) {
            // 1D array of vector.
            SC_Value result = f.sc_.newvalue(SC_Type::Num());
            f.sc_.out() << "  " << result.type << " " << result << " = "
                << array
                << "[int(" << ix1 << ")]"
                << "[int(" << ix2 << ")];\n";
            return result;
        }
    }
    throw Exception(acx, "2 indexes (a[i,j]) not supported for this array");
}

// compile array[i,j,k] expression
SC_Value sc_eval_index3_expr(
    SC_Value array, Operation& op_ix1, Operation& op_ix2, Operation& op_ix3,
    SC_Frame& f, const Context& acx)
{
    if (array.type.plex_array_rank() == 2
        && array.type.plex_array_base().is_vec())
    {
        // 2D array of vector.
        auto ix1 = sc_eval_expr(f, op_ix1, SC_Type::Num());
        auto ix2 = sc_eval_expr(f, op_ix2, SC_Type::Num());
        auto ix3 = sc_eval_expr(f, op_ix3, SC_Type::Num());
        SC_Value result =
            f.sc_.newvalue(array.type.plex_array_base().elem_type());
        f.sc_.out() << "  " << result.type << " " << result << " = " << array
                 << "[int(" << ix1 << ")*" << array.type.plex_array_dim(1)
                 << "+" << "int(" << ix2 << ")][int(" << ix3 << ")];\n";
        return result;
    }
    throw Exception(acx, "3 indexes (a[i,j,k]) not supported for this array");
}

SC_Value Call_Expr::sc_eval(SC_Frame& f) const
{
    SC_Value scval;
    if (sc_try_eval(*func_, f, scval)) {
        if (!scval.type.is_list())
            throw Exception(At_SC_Phrase(func_->syntax_, f),
                stringify("type ", scval.type, ": not an array or function"));
        auto list = cast<List_Expr>(arg_);
        if (list == nullptr)
            throw Exception(At_SC_Phrase(arg_->syntax_, f),
                "expected '[index]' expression");
        if (list->size() == 1)
            return sc_eval_index_expr(scval, *list->at(0), f);
        if (list->size() == 2)
            return sc_eval_index2_expr(scval, *list->at(0), *list->at(1), f,
                At_SC_Phrase(arg_->syntax_, f));
        if (list->size() == 3)
            return sc_eval_index3_expr(scval,
                *list->at(0), *list->at(1), *list->at(2),
                f, At_SC_Phrase(arg_->syntax_, f));
    }
    Value val = sc_constify(*func_, f);
    if (auto func = maybe_function(val, At_SC_Phrase(func_->syntax_,f))) {
        return func->sc_call_expr(*arg_, syntax_, f);
    }
    throw Exception(At_SC_Phrase(func_->syntax_, f),
        stringify("",val," is not an array or function"));
}
SC_Value Index_Expr::sc_eval(SC_Frame& f) const
{
    SC_Value scval = sc_eval_op(f, *arg1_);
    if (!scval.type.is_list())
        throw Exception(At_SC_Phrase(arg1_->syntax_, f),
            stringify("type ", scval.type, ": not an array"));
    return sc_eval_index_expr(scval, *arg2_, f);
}
SC_Value Slice_Expr::sc_eval(SC_Frame& f) const
{
    SC_Value scval = sc_eval_op(f, *arg1_);
    if (!scval.type.is_list())
        throw Exception(At_SC_Phrase(arg1_->syntax_, f),
            stringify("type ", scval.type, ": not an array"));
    auto list = cast<List_Expr>(arg2_);
    if (list == nullptr)
        throw Exception(At_SC_Phrase(arg2_->syntax_, f),
            "expected '[index]' expression");
    if (list->size() == 1)
        return sc_eval_index_expr(scval, *list->at(0), f);
    if (list->size() == 2)
        return sc_eval_index2_expr(scval, *list->at(0), *list->at(1), f,
            At_SC_Phrase(arg2_->syntax_, f));
    if (list->size() == 3)
        return sc_eval_index3_expr(scval,
            *list->at(0), *list->at(1), *list->at(2),
            f, At_SC_Phrase(arg2_->syntax_, f));
    throw Exception(At_SC_Phrase(arg2_->syntax_, f),
        stringify("index list has ",list->size()," components: "
            "only 1..3 supported"));
}

SC_Value Local_Data_Ref::sc_eval(SC_Frame& f) const
{
    return f[slot_];
}

SC_Value Nonlocal_Data_Ref::sc_eval(SC_Frame& f) const
{
    return sc_eval_const(f, f.nonlocals_->at(slot_), *syntax_);
}
SC_Value Symbolic_Ref::sc_eval(SC_Frame& f) const
{
    auto b = f.nonlocals_->dictionary_->find(name_);
    assert(b != f.nonlocals_->dictionary_->end());
    Value val = f.nonlocals_->get(b->second);
    return sc_eval_const(f, val, *syntax_);
}

SC_Value List_Expr_Base::sc_eval(SC_Frame& f) const
{
    if (this->size() >= 2 && this->size() <= 4) {
        SC_Value elem[4];
        for (unsigned i = 0; i < this->size(); ++i) {
            elem[i] = sc_eval_op(f, *this->at(i));
            SC_Type etype = elem[i].type;
            if (!etype.is_num() && !etype.is_bool() && !etype.is_bool32()
                && !etype.is_num_vec())
            {
                throw Exception(At_SC_Phrase(this->at(0)->syntax_, f),
                    stringify(
                        "vector elements must be Num, Bool, Bool32 or Num_Vec;"
                        " got type: ",etype));
            }
            if (i > 0 && etype != elem[0].type) {
                throw Exception(At_SC_Phrase(this->at(i)->syntax_, f),
                    stringify(
                        "vector elements must have uniform type;"
                        " got types ",elem[0].type," and ",etype));
            }
        }
        SC_Type atype = SC_Type::List(elem[0].type, this->size());
        SC_Value result = f.sc_.newvalue(atype);
        f.sc_.out() << "  " << atype << " " << result << " = " << atype << "(";
        bool first = true;
        for (unsigned i = 0; i < this->size(); ++i) {
            if (!first) f.sc_.out() << ",";
            first = false;
            f.sc_.out() << elem[i];
        }
        f.sc_.out() << ");\n";
        return result;
    }
    Value val = sc_constify(*this, f);
    return sc_eval_const(f, val, *syntax_);
}

SC_Value Or_Expr::sc_eval(SC_Frame& f) const
{
    // TODO: change Or to use lazy evaluation.
    auto arg1 = sc_eval_expr(f, *arg1_, SC_Type::Bool());
    auto arg2 = sc_eval_expr(f, *arg2_, SC_Type::Bool());
    SC_Value result = f.sc_.newvalue(SC_Type::Bool());
    f.sc_.out() <<"  bool "<<result<<" =("<<arg1<<" || "<<arg2<<");\n";
    return result;
}
SC_Value And_Expr::sc_eval(SC_Frame& f) const
{
    // TODO: change And to use lazy evaluation.
    auto arg1 = sc_eval_expr(f, *arg1_, SC_Type::Bool());
    auto arg2 = sc_eval_expr(f, *arg2_, SC_Type::Bool());
    SC_Value result = f.sc_.newvalue(SC_Type::Bool());
    f.sc_.out() <<"  bool "<<result<<" =("<<arg1<<" && "<<arg2<<");\n";
    return result;
}
SC_Value If_Else_Op::sc_eval(SC_Frame& f) const
{
    // TODO: change If to use lazy evaluation.
    auto arg1 = sc_eval_expr(f, *arg1_, SC_Type::Bool());
    auto arg2 = sc_eval_op(f, *arg2_);
    auto arg3 = sc_eval_op(f, *arg3_);
    if (arg2.type != arg3.type) {
        throw Exception(At_SC_Phrase(syntax_, f), stringify(
            "if: type mismatch in 'then' and 'else' arms (",
            arg2.type, ",", arg3.type, ")"));
    }
    SC_Value result = f.sc_.newvalue(arg2.type);
    f.sc_.out() <<"  "<<arg2.type<<" "<<result
             <<" =("<<arg1<<" ? "<<arg2<<" : "<<arg3<<");\n";
    return result;
}
void If_Else_Op::sc_exec(SC_Frame& f) const
{
    auto arg1 = sc_eval_expr(f, *arg1_, SC_Type::Bool());
    f.sc_.out() << "  if ("<<arg1<<") {\n";
    arg2_->sc_exec(f);
    f.sc_.out() << "  } else {\n";
    arg3_->sc_exec(f);
    f.sc_.out() << "  }\n";
}
void If_Op::sc_exec(SC_Frame& f) const
{
    auto arg1 = sc_eval_expr(f, *arg1_, SC_Type::Bool());
    f.sc_.out() << "  if ("<<arg1<<") {\n";
    arg2_->sc_exec(f);
    f.sc_.out() << "  }\n";
}
void While_Op::sc_exec(SC_Frame& f) const
{
    f.sc_.opcaches_.emplace_back(Op_Cache{});
    f.sc_.out() << "  while (true) {\n";
    auto cond = sc_eval_expr(f, *cond_, SC_Type::Bool());
    f.sc_.out() << "  if (!"<<cond<<") break;\n";
    body_->sc_exec(f);
    f.sc_.out() << "  }\n";
    f.sc_.opcaches_.pop_back();
}
void For_Op::sc_exec(SC_Frame& f) const
{
  #define RANGE_EXPRESSIONS 1
    auto range = cast<const Range_Expr>(list_);
    if (range == nullptr)
        throw Exception(At_SC_Phrase(list_->syntax_, f),
            "not a range");
  #if RANGE_EXPRESSIONS
    // range arguments are general expressions
    auto first = sc_eval_expr(f, *range->arg1_, SC_Type::Num());
    auto last = sc_eval_expr(f, *range->arg2_, SC_Type::Num());
    auto step = range->arg3_ != nullptr
        ? sc_eval_expr(f, *range->arg3_, SC_Type::Num())
        : sc_eval_const(f, Value{1.0}, *syntax_);
  #else
    // range arguments are constants
    double first = sc_constify(*range->arg1_, f)
        .to_num(At_SC_Phrase(range->arg1_->syntax_, f));
    double last = sc_constify(*range->arg2_, f)
        .to_num(At_SC_Phrase(range->arg2_->syntax_, f));
    double step = range->arg3_ != nullptr
        ? sc_constify(*range->arg3_, f)
          .to_num(At_SC_Phrase(range->arg3_->syntax_, f))
        : 1.0;
  #endif
    auto i = f.sc_.newvalue(SC_Type::Num());
    f.sc_.opcaches_.emplace_back(Op_Cache{});
  #if RANGE_EXPRESSIONS
    f.sc_.out() << "  for (float " << i << "=" << first << ";"
             << i << (range->half_open_ ? "<" : "<=") << last << ";"
             << i << "+=" << step << ") {\n";
  #else
    f.sc_.out() << "  for (float " << i << "=" << dfmt(first, dfmt::EXPR) << ";"
             << i << (range->half_open_ ? "<" : "<=") << dfmt(last, dfmt::EXPR) << ";"
             << i << "+=" << dfmt(step, dfmt::EXPR) << ") {\n";
  #endif
    pattern_->sc_exec(i, At_SC_Phrase(list_->syntax_, f), f);
    if (cond_) {
        auto cond = sc_eval_expr(f, *cond_, SC_Type::Bool());
        f.sc_.out() << "  if (!"<<cond<<") break;\n";
    }
    body_->sc_exec(f);
    f.sc_.out() << "  }\n";
    f.sc_.opcaches_.pop_back();
}

SC_Value sc_vec_element(SC_Frame& f, SC_Value vec, int i)
{
    SC_Value r = f.sc_.newvalue(vec.type.elem_type());

    if (f.sc_.target_ == SC_Target::glsl && vec.type.is_num_vec()) {
        //use gl_index_letters instread of indices if num vec types are used.
        const char* arg2 = nullptr;
        if (i == 0.0)
            arg2 = ".x";
        else if (i == 1.0)
            arg2 = ".y";
        else if (i == 2.0 && vec.type.count() > 2)
            arg2 = ".z";
        else if (i == 3.0 && vec.type.count() > 3)
            arg2 = ".w";
        if (arg2 == nullptr)
            return r;

        f.sc_.out() << "  " << r.type << " " << r << " = "
            << vec << arg2 << ";\n";
    } else {
        f.sc_.out() << "  " << r.type << " " << r << " = "
            << vec << "[" << i << "];\n";
    }
    return r;
}

SC_Value sc_binop(
    SC_Frame& f, SC_Type rtype, SC_Value x, const char* op, SC_Value y)
{
    auto result = f.sc_.newvalue(rtype);
    f.sc_.out() << "  " << rtype << " " << result << " = "
        << x << op << y << ";\n";
    return result;
}

SC_Value sc_bincall(
    SC_Frame& f, SC_Type rtype, const char* fn, SC_Value x, SC_Value y)
{
    auto result = f.sc_.newvalue(rtype);
    f.sc_.out() << "  " << rtype << " " << result << " = "
        << fn << "(" << x << "," << y << ");\n";
    return result;
}

SC_Value sc_unary_call(SC_Frame& f, SC_Type rtype, const char* fn, SC_Value x)
{
    auto result = f.sc_.newvalue(rtype);
    f.sc_.out() << "  " << rtype << " " << result << " = "
        << fn << "(" << x << ");\n";
    return result;
}

void SC_Value_Expr::exec(Frame& f, Executor&) const
{
    throw Exception(At_Phrase(*syntax_, f),
        "SC_Value_Expr::exec internal error");
}
SC_Value SC_Value_Expr::sc_eval(SC_Frame&) const
{
    return val_;
}

} // namespace curv
