from model import Model

def run_predictor_with_params(params: dict, input_file: str) -> float:
    model_params_list = [
        params['pc_lut_addr_size'],
        params['lhr_lut_addr_size'],
        params['ghr_lut_addr_size'],
        params['ga_lut_addr_size'],
        params['xor_lut_addr_size'],
        params['ghr_size'],
        params['ga_branches'],
        params['weight_adjustment_rate'],
        params['min_weight'],
        params['max_weight'],
    ]

    predictor = Model(model_params_list) 

    num_branches = 0
    num_predicted = 0
    interval = 2000
    max_lines = 15000
    with open(input_file, "r") as f:
        for line in f:
            if num_branches > max_lines:
                break

            pc, outcome = map(int, line.strip().split())
            num_branches += 1
            if predictor.predict_and_train(pc, outcome):
                num_predicted += 1
            # descomente para ativar o bleaching
            #if num_branches % interval == 0:
            #    predictor.apply_bleaching()

    if num_branches == 0:
        return 0.0 
    return (num_predicted / num_branches) * 100.0 

def fitness_function(individual_params: dict, input_file: str) -> float:
    accuracy = run_predictor_with_params(individual_params, input_file)
    return accuracy