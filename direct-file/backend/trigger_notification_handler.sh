#!/usr/bin/env bash

# 1. Grab the JSON file
echo $1
json_file=$1
json_data=$(jq '.' "$json_file")

# 2. Write it to localstack
echo "Copying json file to local stack"
docker cp $json_file localstack:/opt/code/localstack

# 3. Docker exec localstack it to s3
echo "Copying json file to operations bucket"
docker exec -d localstack awslocal s3api put-object --bucket operations-jobs --key $json_file --body $json_file

# 4. Kick off sqs
echo "Publishing Notification Handler event message to SQS"
docker exec -d localstack awslocal sqs send-message --queue-url http://localhost:4566/000000000000/s3-notification-event-queue --message-body "{\"path\" : \"${json_file}\"}"