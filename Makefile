
ENV_NAME=MLOpsProject

CA=source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}

setup:
	conda create -n ${ENV_NAME} python==3.8 -y

	bash -c "${CA} && conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y && pip install -r requirements.txt"

	cp pre_commit .git/hooks
	chmod +x .git/hooks/pre_commit

test:
	bash -c "${CA} && python -m unittest -f"

prefect_deploy_main:
	prefect deployment apply main-deployment.yaml
	prefect deployment run main/main
