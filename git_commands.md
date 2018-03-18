### Cloning a Github
    git clone <git_path>

### Initiating git
    git init

### Syncing data from github
    git pull

### Syncing data to github
    git push

### Adding/Stage files
    git add <files>

### Commiting changes
    git commit -m "msg"

### Reverting any specific file to specific commit point
    git checkout <commit hash> <filename>

### Remove last changes and Sync with remote
git reset --hard
git pull

### stash changes asided, pull and them apply stashed data
git stash
git pull
git stash pop
