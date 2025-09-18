Como testar o código? =D

Comando para teste das funções básica dos pesos dinâmicos, simular algumas predições e mostrar a evolução dos pesos.
```bash
cd tournament
python run_tests.py
```
Comando para teste com dados reais de branch prediction, vai mostrar evolução da precisão e exibir os pesos finais
```bash
python test_with_dataset.py ../Dataset_pc_decimal/I1.txt 1000
```

Comando para comparar pesos fixos contra pesos dinâmicos e mostrar a melhoria da precisão
```bash
python compare_weights.py ../Dataset_pc_decimal/I1.txt 1000
```

