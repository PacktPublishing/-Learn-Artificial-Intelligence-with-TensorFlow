// File: section5/load_saved_model.cc
// Author: Brandon McKinzie
// Description: Simple illustration of loading a model from a
//  an export directory (originally a timestamp directory), and
//  feeding it some simple inputs to retrieve predictions back.

#include <memory>
#include <vector>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

std::string REPO_ROOT{"/home/brandon/Documents/ai-with-tensorflow"};

using std::vector;
using std::pair;
using std::string;

using tensorflow::RunOptions;
using tensorflow::SessionOptions;
using tensorflow::Tensor;
using tensorflow::TensorShape;

using tensorflow::DT_STRING;
using tensorflow::LoadSavedModel;
using tensorflow::SavedModelBundle;

int main() {
    // SavedModelBundle objects are what we use
    // to wrap our model (MetaGraphDef) and Session
    // together when working with the C++ API.
    auto bundle = std::make_shared<SavedModelBundle>();
    // Specify path to export_dir, which is the directory that contains:
    // - assets directory
    // - variables directory
    // - saved_mode.pb file
    std::string modelPath{REPO_ROOT + "/section5/export_dir"};
    // Create dummy placeholder objects necessary to call LoadSavedModel.
    // In more complex use cases, we could set the attributes of
    // these to customize our session and run options as needed.
    SessionOptions sessionOptions;
    RunOptions runOptions;
    // Load the SavedModelBundle. We wrap the call with TF_CHECK_OK
    // to ensure LoadSavedModel returned with status.ok(),
    // otherwise the program would crash here.
    TF_CHECK_OK(LoadSavedModel(sessionOptions, runOptions, modelPath,
                               {tensorflow::kSavedModelTagServe},
                               bundle.get()));

    // Create a simple sequence of words to feed to the model.
    // Usually we'd accept a single string from the caller/client
    // and split the sequence into a vector or matrix.
    vector<string> words = {
        "i", "like", "dogs", "they", "are", "cool"
    };

    // Fill a Tensor object with our simple input sequence.
    // It will have shape (1, numWords), where the 1 represents
    // the batch size, which is 1 since we are feeding a single
    // sentence. Our sentence consists of numWords words.
    const int numWords = static_cast<int>(words.size());
    Tensor wordsTensor(DT_STRING, TensorShape({1, numWords}));
    // To fill a Tensor with our words, we have to first
    // get a mutable reference to its flattened data container.
    // Then, we copy our words vector (which is already flattened)
    // into that reference. Welcome to C++!
    auto tensorData = wordsTensor.flat<std::string>().data();
    copy_n(words.begin(), numWords, tensorData);

    // Define what we usually call the feed_dict object to be fed
    // to our session run call. We make use of the implicit conversion
    // abilities of tensorflow::FeedType from vector<pair<string, Tensor>>.
    vector<pair<string, Tensor>> feedInputs = {{"x", wordsTensor}};
    // Create the `outputs` vector that will get filled with
    // each tensor given by `fetches`, which for us is just the
    // predicted category, aka 'preds_words:0'.
    vector<Tensor> outputs;
    vector<string> fetches = {"preds_words:0"};
    // Execute the session run call. The equivalent python code
    // for this would be:
    //  outputs = sess.run('preds_words:0', feed_dict={'x', wordsTensor})
    // Run returns a tensorflow Status object, which has a
    // boolean ok() method, which we can call to check that
    // the graph execution completed without errors.
    auto status = bundle->session->Run(feedInputs, fetches, {}, &outputs);

    // Simple way to check for errors.
    if (!status.ok()) {
        LOG(ERROR) << status.error_message();
        return -1;
    }

    // Display the result.
    // Tensor objects store their data using the C++ Eigen library,
    // and things like .vec<std::string>() are Eigen syntax.
    auto predsWords = outputs[0].vec<std::string>();
    LOG(INFO) << "PredsWords has " << outputs[0].NumElements() << " elements.";
    // The simplest way to access the individual data elements is
    // by iterating NumElements() times and using the object's
    // operator(i) for accessing the i'th element.
    for (int i = 0; i < outputs[0].NumElements(); i++) {
        LOG(INFO) << "Category: " << predsWords(i);
    }

    return 0;

}
