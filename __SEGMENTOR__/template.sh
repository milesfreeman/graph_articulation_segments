nvidia-docker run  -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.13.1-gpu-py3 bash
pip3  install scipy
pip3  install progressbar
python3 ./1_PREDICT.py STG_DATA.pkl
exit
python3 ./2_PROCESS.py raw/temp/TMP.pkl
rm -f raw/temp/TMP.pkl
