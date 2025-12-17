#!/usr/bin/env fish

# Script pour créer et activer l'environnement conda pour les scripts create_dataset et gaussian_noise

echo "Création de l'environnement conda 'choquet_env'..."

# Créer l'environnement à partir du fichier environment.yml
conda env create -f environment.yml

if test $status -eq 0
    echo "Environnement 'choquet_env' créé avec succès!"
    echo ""
    echo "Pour activer l'environnement, utilisez:"
    echo "conda activate choquet_env"
    echo ""
    echo "Pour tester l'environnement, vous pouvez lancer:"
    echo "python test_conda_environment.py"
else
    echo "Erreur lors de la création de l'environnement!"
    exit 1
end
