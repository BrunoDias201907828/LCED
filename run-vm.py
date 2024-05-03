import argparse
import os
import subprocess
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the VM')
    parser.add_argument('-u', '--user', type=str, required=True, help='username')
    args = parser.parse_args()

    command = f"""
    set -e
    
    rsync -av -e ssh ./ lced1@10.227.243.131:/home/lced1/code/user/{args.user} \
      --exclude='venv/' \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='.idea' \
      --exclude='*.pdf' \
      --exclude='*.csv'
    ssh -tt -L 8888:localhost:8888 lced1@10.227.243.131 "cd ~/code/user/{args.user}/LCED; ./docker-run.sh '$*'; exec bash;"
    """

    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        tmp.write(command)
        os.system(f"chmod +x {tmp.name}")
        os.system(f"{tmp.name}")
