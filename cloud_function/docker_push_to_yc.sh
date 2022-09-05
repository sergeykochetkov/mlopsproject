#!/bin/bash

set -x

tag=${1} &&

new_tag=cr.yandex/crprgafub8l75q50vui3/$tag &&

docker tag $tag $new_tag &&

docker push $new_tag &&

echo image ${tag} succesefully pushed to ${new_tag}

