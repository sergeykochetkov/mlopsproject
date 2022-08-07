
ENV_NAME=MLOpsProject

setup:
	conda create -n ${ENV_NAME} python==3.8 -y

	bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME} && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y && pip install -r requirements.txt"

