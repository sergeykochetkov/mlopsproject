
ENV_NAME=MLOpsProject

CA=source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}

install_miniconda:
	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm -rf ~/miniconda3/miniconda.sh
	~/miniconda3/bin/conda init bash
	~/miniconda3/bin/conda init zsh

setup:
	conda create -n ${ENV_NAME} python==3.8 -y

	bash -c "${CA} && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y && pip install -r requirements.txt"

	cp pre-commit .git/hooks
	chmod +x .git/hooks/pre-commit

test:
	bash -c "${CA} && python -m unittest -f"

prefect_deploy_main:
	prefect deployment apply main-deployment.yaml
	prefect deployment run main/main
