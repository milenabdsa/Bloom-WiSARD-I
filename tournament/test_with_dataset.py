#!/usr/bin/env python3
#Por favor NÃO REMOVER O SHEBANG!!!
#Criei esse lindo arquivinho para rodar testes com datasets reais

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model

def test_with_dataset(dataset_path, num_branches=10000):
    print(f"=== Testando com dataset: {dataset_path} ===")
    
    parameters = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8000, 8000, 8000, 8000, 8000, 5.0, 2.0, 2.0, 2.0, 3.0, 3, 3, 3, 3, 3, 4, 4]

    parameters = {
        "M1": [22, 31, 58, 29, 63, 519, 32, 65, 696, 1169],
        "M2": [ 26, 29, 16, 29, 20, 56, 231, 1, 1, 2000],
        "I1": [],
        "I2": [],
    }
    
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    if dataset_name not in parameters:
        print(f"Parâmetros não encontrados para o dataset {dataset_name}. Datasets aceitos são {list(parameters.keys())}.")
        return

    model = Model(parameters[dataset_name])
    
    print("Pesos iniciais:", model.get_dynamic_weights())
    
    correct = 0
    total = 0
    
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= num_branches:
                break
                
            pc, outcome = map(int, line.strip().split())
            
            if model.predict_and_train(pc, outcome):
                correct += 1
            total += 1
            
            if total % 10000 == 0:
                accuracy = (correct / total) * 100
                print(f"Branches: {total}, Precisão: {accuracy:.2f}%, Pesos: {model.get_dynamic_weights()}")
    
    final_accuracy = (correct / total) * 100
    print(f"\n=== Resultados Finais ===")
    print(f"Precisão final: {final_accuracy:.2f}%")
    print(f"Pesos finais: {model.get_dynamic_weights()}")
    
    return final_accuracy

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        num_branches = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        test_with_dataset(dataset_path, num_branches)
    else:
        print("Uso: python test_with_dataset.py <caminho_do_dataset> [numero_de_branches]")
        print("Exemplo: python test_with_dataset.py ../Dataset_pc_decimal/I1.txt 5000")

