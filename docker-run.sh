set -e

docker build -t local .
docker run -it --rm \
  -v ./:/git-repos \
  -v /home/lced1/datasets/:/datasets \
  --network=host \
  local
