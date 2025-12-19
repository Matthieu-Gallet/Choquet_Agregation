# Lister seulement les packages que vous avez installés manuellement
python -m pip list --not-required --format=freeze > requirements_minimal.txt



# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Créer un nouvel environnement avec uv
uv venv venv_uv

# Activer le nouvel environnement
source venv_uv/bin/activate

# Installer tous les packages à partir de requirements.txt
uv pip install -r requirements_minimal.txt