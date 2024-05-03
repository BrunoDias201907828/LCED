sshpass -e rsync --no-perms -av -e ssh ~/Projects/LCED/ lced1@10.227.243.131:/home/lced1/code/user/ptavares \
  --exclude='venv/' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.idea' \
  --exclude='*.pdf' \
  --exclude='*.csv'
sshpass -e ssh -tt -L 8888:localhost:8888 lced1@10.227.243.131 "cd ~/code/user/ptavares ; chmod -R 777 . ; ./docker-run.sh '$*'; exec bash;"
