export CKPT=/mnt/sdb/CKPT/VCD
export DATA=/mnt/sdb/DATA/VCD
export PREP=/mnt/sdb/PREP/VCD

export SPLITS=train:val:test

echo Running "source ./starter/env_definer.sh"
echo CKPT: $CKPT
echo PREP: $PREP
echo DATA: $DATA
