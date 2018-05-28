#!/usr/bin/env python3

"""model_client.py: Get predictions from model server given by FLAGS.server.

Pseudocode Overview:

    predict_category:
        X, Y = get_vectorized_data_matrices()
        stub = PredictionService_stub(gRPC_channel)
        for i in range(num_tests):
            request = PredictRequest(...) # proto class


apis/predict.proto:
    message PredictRequest {
        ModelSpec model_spec = 1;
        map<string, TensorProto> inputs = 2;
        repeated string output_filter = 3;
    }
    message PredictResponse {
        map<string, TensorProto> outputs = 1;
    }
    
apis/classification.proto:
    message Class {
      string label = 1;
      float score = 2;
    }
    message Classifications {
      repeated Class classes = 1;
    }
    message ClassificationResult {
      repeated Classifications classifications = 1;
    }
    message ClassificationRequest {
      ModelSpec model_spec = 1;
      tensorflow.serving.Input input = 2;
    }
    message ClassificationResponse {
      ClassificationResult result = 1;
    }

apis/prediction_service.proto:
    service PredictionService {
        rpc Classify(ClassificationRequest) returns (ClassificationResponse);
        rpc Regress(RegressionRequest) returns (RegressionResponse);
        rpc Predict(PredictRequest) returns (PredictResponse);
        rpc MultiInference(MultiInferenceRequest) returns (MultiInferenceResponse);
        rpc GetModelMetadata(GetModelMetadataRequest)
            returns (GetModelMetadataResponse);
    }

Notes:
    The "some_stub.SomeMethod.future(...)" signature looks like:
        (self, request, timeout=None, metadata=None, credentials=None):
        Source: https://grpc.io/grpc/python/_modules/grpc.html
"""

import sys
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, model_pb2, prediction_service_pb2

import argparse
DESCRIPTION = """Make queries to a TensorFlow ModelServer."""
parser = argparse.ArgumentParser(
    description=DESCRIPTION,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    '--server', default='localhost:9000',
    help='PredictionService host:port')


def main():
    args = parser.parse_args()

    if not args.server:
        print('please specify server host:port')
        sys.exit()

    # Get the server stub. This is how we interact with the server.
    host, port = args.server.split(':')
    stub = get_prediction_service_stub(host, int(port))

    # Create test example to send as 'x' for prediction.
    X = np.array(['i', 'like', 'dogs'])
    # Reshape to indicate batch of size 1.
    X = np.reshape(X, (1, 3))

    # Issue predict request to tensorflow model server,
    category = predict_category(stub, X)
    print('Predicted category:', category)


def get_prediction_service_stub(host, port):
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    return stub


def predict_category(stub, X):
    # Wrap X inside a valid PredictRequest.
    predict_request = get_predict_request(X)
    # Call TensorFlow model server's Predict API, which returns a PredictResponse.
    predict_response = stub.Predict(predict_request, timeout=20.0)
    # Extract the predicted category from the PredictResponse object.
    prediction_category = get_predicted_category(predict_response)
    return prediction_category


def get_predict_request(x):
    model_spec = model_pb2.ModelSpec(name='default', signature_name='export_outputs')
    request = predict_pb2.PredictRequest(model_spec=model_spec)
    request.inputs['x'].CopyFrom(
        tf.contrib.util.make_tensor_proto(x, shape=x.shape))
    return request


def get_predicted_category(predict_response):
    return predict_response.outputs['preds_words'].string_val[0].decode()


if __name__ == '__main__':
    main()

