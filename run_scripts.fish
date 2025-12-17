#!/usr/bin/env fish

# Script de lancement facile pour les scripts Choquet Aggregation

set PYTHON_EXEC "/home/mgallet/Documents/Codes/Python/3_DEVELOPPEMENT/Choquet_Agregation/.venv/bin/python"
set BASE_DIR "/home/mgallet/Documents/Codes/Python/3_DEVELOPPEMENT/Choquet_Agregation"

function show_help
    echo "Lanceur de scripts Choquet Aggregation"
    echo ""
    echo "Usage: ./run_scripts.fish [OPTION]"
    echo ""
    echo "Options:"
    echo "  test         - Tester l'environnement"
    echo "  dataset      - Lancer create_dataset.py"
    echo "  noise        - Lancer gaussian_noise_mul.py avec paramètres par défaut"
    echo "  noise [args] - Lancer gaussian_noise_mul.py avec arguments personnalisés"
    echo "  help         - Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  ./run_scripts.fish test"
    echo "  ./run_scripts.fish dataset"
    echo "  ./run_scripts.fish noise"
    echo "  ./run_scripts.fish noise --polution=0.2 --balanced=True --winsize=20"
end

function test_environment
    echo "Test de l'environnement..."
    cd $BASE_DIR
    $PYTHON_EXEC test_final_environment.py
end

function run_dataset
    echo "Lancement de create_dataset.py..."
    cd $BASE_DIR/cpazmal_analyse
    $PYTHON_EXEC create_dataset.py
end

function run_noise
    echo "Lancement de gaussian_noise_mul.py..."
    cd $BASE_DIR/src
    
    if test (count $argv) -eq 0
        echo "Paramètres par défaut: --polution=0.1 --balanced=True --winsize=15"
        $PYTHON_EXEC gaussian_noise_mul.py --polution=0.1 --balanced=True --winsize=15
    else
        echo "Paramètres personnalisés: $argv"
        $PYTHON_EXEC gaussian_noise_mul.py $argv
    end
end

# Script principal
if test (count $argv) -eq 0
    show_help
    exit 1
end

switch $argv[1]
    case "test"
        test_environment
    case "dataset"
        run_dataset
    case "noise"
        run_noise $argv[2..-1]
    case "help"
        show_help
    case "*"
        echo "Option inconnue: $argv[1]"
        echo ""
        show_help
        exit 1
end
