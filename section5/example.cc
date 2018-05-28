// tensorflow/cc/example/example.cc
// Original code from https://www.tensorflow.org/api_guides/cc/guide, 
// with verbose comments added by me (Brandon McKinzie).
//
// I've also removed the "using namespace" declarations,
// since it makes it confusing for anyone wondering which code came from where.
// I've instead explicitly wrote using declarations for each object used.

// client_session.h gives us both tensorflow::ClientSession and tensorflow::Scope.
// In my opinion, it is rather misleading that TensorFlow didn't explicitly
// #include tensorflow/cc/framework/scope.h
// in this example, since that's how we get Scope, as it is included
// by client_session.h.
#include "tensorflow/cc/client/client_session.h"

// standard_ops.h is simply a list of #include directives to various
// files inside tensorflow/core/ops.
// Note that standard_ops.h includes these as if they were inside
// tensorflow/cc/ops, although the majority clearly aren't, which
// is confusing to say the least. The details aren't important,
// but it's because the build rule for standard_ops.h (cc_ops) uses a
// custom bazel function called tf_gen_op_wrappers_cc. You can see
// the details by looking through tensorflow/cc/BUILD,
// which contains the tf_gen_op_wrappers_cc(name="cc_ops",...)
// build rule. The function tf_gen_op_wrappers_cc itself is defined in
// tensorflow/tensorflow.bzl.
//
// All that really matters, though, is that you include this header whenever you
// want to use stuff in the tensorflow::ops namespace. You'll know if something is 
// in this namespace because it will say so in its documentation under the C++ API
// on tensorflow.org.
#include "tensorflow/cc/ops/standard_ops.h"

// tensorflow::Tensor
#include "tensorflow/core/framework/tensor.h"

using tensorflow::Scope;
using tensorflow::Output;
using tensorflow::Tensor;
using tensorflow::ClientSession;

using tensorflow::ops::Const;
using tensorflow::ops::MatMul;

int main() {

    // tensorflow::Scope is the main data structure that holds the
    // current state of graph construction. A Scope acts as a handle
    // to the graph being constructed, as well as storing TensorFlow
    // operation properties.
    //
    // The Scope object is the first argument to operation constructors,
    // and operations that use a given Scope as their first
    // argument inherit that Scope's properties, such as a common
    // name prefix.
    //
    // NewRootScope creates some resources such as a graph to which
    // operations are added. It also creates a tensorflow::Status object
    // which will be used to indicate errors encountered when constructing
    // operations. The Scope class has value semantics, thus, a Scope
    // object can be freely copied and passed around.
    // The Scope object returned by Scope::NewRootScope is referred to
    // as the root scope. "Child" scopes can be constructed from the
    // root scope by calling various member functions of the Scope class,
    // thus forming a hierarchy of scopes.
    Scope root = Scope::NewRootScope();

    // Matrix A = [3 2; -1 0]
    Output A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
    // Vector b = [3 5]
    Output b = Const(root, { {3.f, 5.f} });
    // v = Ab^T
    auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));

    // Run the C++ equivalant of `outputs = sess.run(v)` (Python).
    // Note: TF_CHECK_OK and other TF macros are defined in
    // tensorflow/core/lib/core/status.h.
    std::vector<Tensor> outputs;
    ClientSession session(root);
    TF_CHECK_OK(session.Run({v}, &outputs));

    // Expect outputs[0] == [19; -3]
    LOG(INFO) << outputs[0].matrix<float>();
    return 0;
}
