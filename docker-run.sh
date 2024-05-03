set -e

docker build -t local .
docker run -it --rm \
  -v ./:/git-repos \
  --network=host \
  local
