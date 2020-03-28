nvidia-docker run  -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.13.1-gpu-py3 bash
pip3  install scipy
pip3  install progressbar
python3 ./deploy.py 
exit
python3 ./process_pred.py temp.pkl False
rm -f temp.pkl