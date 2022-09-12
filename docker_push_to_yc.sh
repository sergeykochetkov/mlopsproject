#!/bin/bash

set -x

YC_CONTAINER_REGISTRY=cr.yandex/crprgafub8l75q50vui3

tag=${1} &&

docker build -t $tag -f serverless_container.Dockerfile . &&

new_tag=$YC_CONTAINER_REGISTRY/$tag &&

docker tag $tag $new_tag &&

docker push $new_tag &&

echo image ${tag} succesefully pushed to ${new_tag}

