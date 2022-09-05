#!/bin/bash

set -x

QUEUE_URL="https://message-queue.api.cloud.yandex.net/b1gkgpqrk3vp6tfjqpbn/dj6000000006h57m06dk/test-queue"

END_POINT="--endpoint https://message-queue.api.cloud.yandex.net/"

for ((i=0;i<10;i++))
do
aws sqs send-message --message-body "Hello World_$i" $END_POINT --queue-url $QUEUE_URL
done

aws sqs receive-message $END_POINT --queue-url $QUEUE_URL

echo $?
