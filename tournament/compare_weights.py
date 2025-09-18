#!/usr/bin/env python3
#Por favor NÃO REMOVER O SHEBANG!!!
#Criei esse lindo arquivinho para comparar pesos fixos vs pesos dinâmicos

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model

class OriginalModel(Model):
    def __init__(self, input_params):
        super().__init__(input_params)
        self.use_dynamic_weights = False
    
    def _tournament_predict(self, pc_count_0, pc_count_1, xor_count_0, xor_count_1, 
                           lhr_count_0, lhr_count_1, ghr_count_0, ghr_count_1, 
                           ga_count_0, ga_count_1):
        if self.use_dynamic_weights:
            return super()._tournament_predict(pc_count_0, pc_count_1, xor_count_0, xor_count_1,
                                             lhr_count_0, lhr_count_1, ghr_count_0, ghr_count_1,
                                             ga_count_0, ga_count_1)
        else:
            overall_count_0 = (self.pc_tournament_weight * pc_count_0 + 
                              self.lhr_tournament_weight * lhr_count_0 + 
                              self.ghr_tournament_weight * ghr_count_0 + 
                              self.ga_tournament_weight * ga_count_0 + 
                              self.xor_tournament_weight * xor_count_0)
            
            overall_count_1 = (self.pc_tournament_weight * pc_count_1 + 
                              self.lhr_tournament_weight * lhr_count_1 + 
                              self.ghr_tournament_weight * ghr_count_1 + 
                              self.ga_tournament_weight * ga_count_1 + 
                              self.xor_tournament_weight * xor_count_1)
            
            return 0 if overall_count_0 > overall_count_1 else 1

def compare_models(dataset_path, num_branches=10000):
    print(f"=== Comparando Pesos Fixos vs Dinâmicos ===")
    print(f"Dataset: {dataset_path}")
    print(f"Branches: {num_branches}")
    
    parameters = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8000, 8000, 8000, 8000, 8000, 5.0, 2.0, 2.0, 2.0, 3.0, 3, 3, 3, 3, 3, 4, 4]
    
    model_fixed = OriginalModel(parameters)
    model_fixed.use_dynamic_weights = False
    
    model_dynamic = OriginalModel(parameters)
    model_dynamic.use_dynamic_weights = True
    
    correct_fixed = 0
    correct_dynamic = 0
    total = 0
    
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= num_branches:
                break
                
            pc, outcome = map(int, line.strip().split())
            
            if model_fixed.predict_and_train(pc, outcome):
                correct_fixed += 1
            if model_dynamic.predict_and_train(pc, outcome):
                correct_dynamic += 1
            total += 1
    
    accuracy_fixed = (correct_fixed / total) * 100
    accuracy_dynamic = (correct_dynamic / total) * 100
    
    print(f"\n=== Resultados ===")
    print(f"Pesos Fixos: {accuracy_fixed:.2f}%")
    print(f"Pesos Dinâmicos: {accuracy_dynamic:.2f}%")
    print(f"Melhoria: {accuracy_dynamic - accuracy_fixed:+.2f}%")
    print(f"Pesos finais dinâmicos: {model_dynamic.get_dynamic_weights()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        num_branches = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
        compare_models(dataset_path, num_branches)
    else:
        print("Uso: python compare_weights.py <caminho_do_dataset> [numero_de_branches]")
        print("Exemplo: python compare_weights.py ../Dataset_pc_decimal/I1.txt 5000")

