sshpass -e rsync -av -e ssh ~/Projects/LCED/ lced1@10.227.243.131:/home/lced1/code/user/ptavares \
  --exclude='venv/' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.idea' \
  --exclude='*.pdf' \
  --exclude='*.csv'
sshpass -e ssh -tt -L 5000:localhost:5000 lced1@10.227.243.131 "cd ~/code/user/ptavares; ./docker-run.sh '$*'; exec bash;"
