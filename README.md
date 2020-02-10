# Chest X Ray Modelling with Deep Learning

This project is executed two phases:

## Phase 1  - Model Building

A deep learning model is trained which trains a model using transfer learning in an attempt to identify one or more existing conditions existing in a Chest X Ray.

Following file builds the model with a smaller dataset.
https://github.com/ERS-HCL/chest-x-ray-model-deep-learning/blob/master/chest-xray-modelling-version-1.ipynb

To include all of the data in the model, please refer to following file.
https://github.com/ERS-HCL/chest-x-ray-model-deep-learning/blob/master/chest-xray-modelling-v3-full.ipynb

This model is then exported to a .tar.gz file which is also made available in the repository.


## Phase 2 - Model Hosting

The machine learning model is hosted on an AWS Lambda environment. Source code is available under directory lambda-multilabel.
The tar.gz file is put in S3 Bucket, at the time of deployment, this information is passed to sam cli.

For multilabel processing, one change is required to be done in the following file (app.py). The threshold levels need to be changed. 
thresholds  = torch.tensor([.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20])   # input

https://github.com/ERS-HCL/chest-x-ray-model-deep-learning/blob/master/lambda-multilabel/pytorch/app.py

### Commands to deploy Lambda function on AWS

Trarverse to the directory where template.yaml file is located.

sam package --output-template-file packaged.yaml --s3-bucket landuse-model-bucket

sam deploy --template-file packaged.yaml --stack-name chestxray-app --capabilities CAPABILITY_IAM --parameter-overrides BucketName= ObjectKey=chestxray-v2.tar.gz

### API endpoint: https://k8x8323esc.execute-api.us-east-2.amazonaws.com/Prod/invocations

### Test Command 

curl -d "{\"url\":\"https://landuse-model-bucket.s3.us-east-2.amazonaws.com/CX00028173_004.png\"}" -H "Content-Type: application/json" -X POST https://k8x8323esc.execute-api.us-east-2.amazonaws.com/Prod/invocations
