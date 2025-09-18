#!/usr/bin/env python3
#Por favor NÃO REMOVER O SHEBANG!!!
#Criei esse lindo arquivinho para rodar testes mais completinhos e "robustos" dos pesos dinâmicos

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model

def test_dynamic_weights():
    
    print("=== Teste dos Pesos Dinâmicos ===")
    
    parameters = [
        1,
        1,
        1,
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        8000,
        8000,
        8000,
        8000,
        8000,
        5.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3,
        3,
        3,
        3,
        3,
        4,
        4
    ]
    
    model = Model(parameters)
    
    print("Pesos iniciais:")
    initial_weights = model.get_dynamic_weights()
    for name, weight in initial_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    test_data = [
        (0x1000, 1),
        (0x1004, 0),
        (0x1008, 1),
        (0x100c, 1),
        (0x1010, 0),
    ]
    
    print("\n=== Simulando predições ===")
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for i, (pc, outcome) in enumerate(test_data):
        print(f"\nPredição {i+1}: PC=0x{pc:x}, Outcome={outcome}")
        
        is_correct = model.predict_and_train(pc, outcome)
        
        if is_correct:
            correct_predictions += 1
            print(f"  ✓ Predição correta!")
        else:
            print(f"  ✗ Predição incorreta")
        
        current_weights = model.get_dynamic_weights()
        print("  Pesos atuais:")
        for name, weight in current_weights.items():
            print(f"    {name}: {weight:.3f}")
    
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n=== Resultados ===")
    print(f"Precisão: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
    
    final_weights = model.get_dynamic_weights()
    print("\nPesos finais:")
    for name, weight in final_weights.items():
        change = weight - initial_weights[name]
        print(f"  {name}: {weight:.3f} (mudança: {change:+.3f})")
    
    print("\n=== Testando reset dos pesos ===")
    model.reset_dynamic_weights()
    reset_weights = model.get_dynamic_weights()
    print("Pesos após reset:")
    for name, weight in reset_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    print("\n=== Teste concluído com sucesso! ===")

if __name__ == "__main__":
    try:
        test_dynamic_weights()
    except Exception as e:
        print(f"Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()

