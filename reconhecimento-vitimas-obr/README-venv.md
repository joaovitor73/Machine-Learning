Ambiente virtual (venv)

Como ativar (Linux / macOS):

source .venv/bin/activate

Como desativar:

deactivate

Gerar/atualizar requirements.txt:

# Ative o venv primeiro
source .venv/bin/activate
# Instale pacotes necessários
pip install -r requirements.txt  # para instalar a partir do arquivo
# Depois de adicionar/remover pacotes
pip freeze > requirements.txt