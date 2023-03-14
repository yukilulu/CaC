source   ~/.bashrc
hash -r
export path=$TMPDIR:$path
source  /usr/local/Anaconda3_202007/anaconda3.sh


conda activate sfda
nvidia-smi

#sh digit.sh
cd /export/home/liyg/code/SFDA

sh train_src_oh.sh
sh train_tar.sh
