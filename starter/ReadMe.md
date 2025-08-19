#Perception Magnifier


put below into `~/.zshrc` and source it
```
alias init_pm='cd /mnt/sdb/REPO/PM; pwd; 
                source ./starter/env_definer.sh;
                conda activate PM'
alias szsh='source ~/.zshrc'
```


run ```init_pm```
export PYTHONPATH="${PYTHONPATH}:/mnt/sdb/REPO/PM/"
