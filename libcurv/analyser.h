// Copyright 2016-2020 Doug Moen
// Licensed under the Apache License, version 2.0
// See accompanying file LICENSE or https://www.apache.org/licenses/LICENSE-2.0

#ifndef LIBCURV_ANALYSER_H
#define LIBCURV_ANALYSER_H

#include <libcurv/meaning.h>
#include <libcurv/builtin.h>

namespace curv {

// Shared analysis state while analysing a source file.
struct File_Analyser
{
    System& system_;

    /// file_frame_ is nullptr, unless we are analysing a source file due to
    /// an evaluation-time call to `file`. It's used by the Exception Context,
    /// to add a stack trace to compile time errors.
    Frame* file_frame_;

    // Have we already emitted a 'deprecated' warning for this topic?
    // Used to prevent an avalanche of warning messages.
    bool var_deprecated_ = false;
    bool paren_empty_list_deprecated_ = false;
    bool paren_list_deprecated_ = false;
    bool not_deprecated_ = false;
    bool dot_string_deprecated_ = false;
    bool string_colon_deprecated_ = false;
    bool where_deprecated_ = false;

    File_Analyser(System& system, Frame* file_frame)
    :
        system_(system),
        file_frame_(file_frame)
    {}

    void deprecate(bool File_Analyser::*, int, const Context&, const String_Ref&);
    static const char dot_string_deprecated_msg[];
};

// Local analysis state that changes when entering a new name-binding scope.
struct Environ
{
    Environ* parent_;
    File_Analyser& analyser_;
    slot_t frame_nslots_;
    slot_t frame_maxslots_;

    // constructor for root environment of a source file
    Environ(File_Analyser& analyser)
    :
        parent_(nullptr),
        analyser_(analyser),
        frame_nslots_(0),
        frame_maxslots_(0)
    {}

    // constructor for child environment. parent is != nullptr.
    Environ(Environ* parent)
    :
        parent_(parent),
        analyser_(parent->analyser_),
        frame_nslots_(0),
        frame_maxslots_(0)
    {}

    slot_t make_slot()
    {
        slot_t slot = frame_nslots_++;
        if (frame_maxslots_ < frame_nslots_)
            frame_maxslots_ = frame_nslots_;
        return slot;
    }

    Shared<Meaning> lookup(const Identifier& id);
    Shared<Locative> lookup_lvar(const Identifier& id, unsigned edepth);
    virtual Shared<Meaning> single_lookup(const Identifier&) = 0;
    virtual Shared<Locative> single_lvar_lookup(const Identifier&);
};

struct Builtin_Environ : public Environ
{
protected:
    const Namespace& names;
public:
    Builtin_Environ(const Namespace& n, File_Analyser& a)
    :
        Environ(a),
        names(n)
    {}
    virtual Shared<Meaning> single_lookup(const Identifier&);
};

// Interp is the second argument of Phrase::analyse().
// It means "interpretation": how to interpret the phrase, relative to the
// environment.
//
// 'edepth' is the number of nested environments surrounding the phrase
// in which an lvar (a local variable on the left side of a := statement)
// can be looked up. The parent phrase computes an edepth for each of its
// subphrases. Ultimately the edepth is passed to Environ::lvar_lookup().
// * If PH is a phrase that binds local variables (let, where, for),
//   the body of PH has PH's edepth + 1.
// * Otherwise, if PH is a phrase with sequential order of evaluation for each
//   of its subphrases (eg, semicolon or do phrase), then the edepth of each
//   subphrase is the same as its parent.
// * The common case is a compound phrase that doesn't have a defined order
//   of evaluation. In this case, the edepth of each subphrase is 0, which means
//   that you can't assign local variables inside that phrase that are defined
//   outside that phrase. If you could do so, then the order of evaluation
//   would be exposed. For example, the + operator is commutative, so A+B is
//   equivalent to B+A, so we don't support assignment inside a plus phrase.
struct Interp
{
private:
    unsigned edepth_ = 0;
    bool is_expr_ = true;
    Interp(unsigned d, bool ie) : edepth_(d), is_expr_(ie) {}
public:
    static Interp expr() { return {0, true}; }
    static Interp stmt(unsigned d = 0) { return {d, false}; }
    int edepth() const { return edepth_; }
    bool is_expr() const { return is_expr_; }
    bool is_stmt() const { return !is_expr_; }
    Interp deepen() const { return {edepth_+1, is_expr_}; }
    Interp to_expr() const { return {edepth_, true}; }
    Interp to_stmt() const { return {edepth_, false}; }
};

// Analyse a Phrase, throw an error if it is not an Operation.
Shared<Operation> analyse_op(const Phrase& ph, Environ& env,
    Interp terp=Interp::expr());

// Evaluate the phrase as a constant expression in the builtin environment.
Value std_eval(const Phrase& ph, Environ& env);

} // namespace
#endif // header guard
