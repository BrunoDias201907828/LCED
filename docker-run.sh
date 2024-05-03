docker build -t local .
docker run -it --rm \
  -v ./:/git-repos \
  local 