# Chest X Ray Modelling with Deep Learning

This project can be seen in two phases:

Phase 1

A deep learning model is trained which trains a model using transfer learning in an attempt to identify one or more existing conditions existing in a Chest X Ray.

This model is then exported to a .tar.gz file which is also made available in the repository.


Phase 2

The machine learning model is hosted on an AWS Lambda environment. This is available under lambda-multilabel.

Commands to deploy Lambda function on AWS

Trarverse to the directory where template.yaml file is located.

sam package --output-template-file packaged.yaml --s3-bucket landuse-model-bucket

sam deploy --template-file packaged.yaml --stack-name chestxray-app --capabilities CAPABILITY_IAM --parameter-overrides BucketName= ObjectKey=chestxray-v2.tar.gz


API endpoint: https://k8x8323esc.execute-api.us-east-2.amazonaws.com/Prod/invocations

## Test Command 

curl -d "{\"url\":\"https://landuse-model-bucket.s3.us-east-2.amazonaws.com/CX00028173_004.png\"}" -H "Content-Type: application/json" -X POST https://k8x8323esc.execute-api.us-east-2.amazonaws.com/Prod/invocations
