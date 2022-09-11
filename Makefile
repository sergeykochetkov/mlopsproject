
ENV_NAME=MLOpsProject

CA0=source ~/anaconda3/etc/profile.d/conda.sh
CA=${CA0} && conda activate ${ENV_NAME}

install_anaconda:
	mkdir -p ~/anaconda3/
	wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda3/anaconda.sh
	bash ~/anaconda3/anaconda.sh -b -u -p ~/anaconda3
	rm -rf ~/anaconda3/anaconda.sh
	~/anaconda3/bin/conda init bash
	~/anaconda3/bin/conda init zsh

create_conda_env:
	bash -c "${CA0} && conda create -n ${ENV_NAME} python==3.8 -y"

setup_hooks:
	bash -c "if [[ -d .git ]]; then cp pre-commit .git/hooks && chmod +x .git/hooks/pre-commit ; fi"

install_requirements_gpu:
	bash -c "${CA} && conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio cudatoolkit=11.3 -c pytorch -y && pip install -r requirements.txt"

install_requirements_cpu:
	bash -c "${CA} && conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio cpuonly -c pytorch -y && pip install -r requirements.txt"

setup: create_conda_env install_requirements_gpu setup_hooks

setup_cpu: create_conda_env install_requirements_cpu setup_hooks

test:
	bash -c "${CA} && python -m unittest -f"

prefect_deploy_main:
	prefect deployment apply main-deployment.yaml
	prefect deployment run main/main

run_integration_test:
	bash -c "./run_integrational_test.sh"

