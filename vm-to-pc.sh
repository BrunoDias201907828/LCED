while getopts ":u:" opt; do
  case ${opt} in
    u )
      username=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# Check if username is provided
if [ -z "$username" ]; then
  echo "Username is required. Please specify using -u option."
  exit 1
fi

rsync -av -e ssh "lced1@10.227.243.131:/home/lced1/code/user/${username}/" ./ \
  --exclude='venv/' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.idea' \
  --exclude='*.pdf' \
  --exclude='*.csv'
