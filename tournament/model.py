from typing import List, Tuple
import numpy as np

from discriminator import Discriminator

class Model:
    def __init__(self, input_params):
        self.num_pc_filters = 1
        self.num_lhr_filters = 1
        self.num_ghr_filters = 1
        self.num_ga_filters = 1
        self.num_xor_filters = 1
        self.pc_lut_addr_size = input_params[0]
        self.lhr_lut_addr_size = input_params[1]
        self.ghr_lut_addr_size = input_params[2]
        self.ga_lut_addr_size = input_params[3]
        self.xor_lut_addr_size = input_params[4]
        self.pc_num_hashes = 3
        self.lhr_num_hashes = 3
        self.ghr_num_hashes = 3
        self.ga_num_hashes = 3
        self.xor_num_hashes = 3
        self.ghr_size = input_params[5]
        self.ga_branches = input_params[6]
        self.weight_adjustment_rate = input_params[7]
        self.min_weight = input_params[8]
        self.max_weight = input_params[9]
        self.seed = 203
        
        self.dynamic_weights = {
            'pc': 1.0,
            'lhr': 1.0,
            'ghr': 1.0,
            'ga': 1.0,
            'xor': 1.0
        }
        
        

        self.pc_discriminators = [
            Discriminator(self.num_pc_filters, self.pc_lut_addr_size, self.pc_num_hashes)
            for _ in range(2)
        ]

        self.xor_discriminators = [
            Discriminator(self.num_xor_filters, self.xor_lut_addr_size, self.xor_num_hashes)
            for _ in range(2)
        ]

        self.lhr_discriminators = [
            Discriminator(self.num_lhr_filters, self.lhr_lut_addr_size, self.lhr_num_hashes)
            for _ in range(2)
        ]

        self.ghr_discriminators = [
            Discriminator(self.num_ghr_filters, self.ghr_lut_addr_size, self.ghr_num_hashes)
            for _ in range(2)
        ]

        self.ga_discriminators = [
            Discriminator(self.num_ga_filters, self.ga_lut_addr_size, self.ga_num_hashes)
            for _ in range(2)
        ]

        #self.ghr_size = 24
        self.ghr = np.zeros(self.ghr_size, dtype=np.uint8)

        self.lhr_configs = [
            (24, 12),  # (comprimento, bits_pc) para LHR1
            (9, 9),  # LHR2
            (5, 5),  # LHR3
        ]

        self.lhrs = []
        for length, bits_pc in self.lhr_configs:
            lhr_size = 1 << bits_pc
            self.lhrs.append(np.zeros((lhr_size, length), dtype=np.uint8))

        self.ga_lower = 8
        self.ga = np.zeros(self.ga_lower * self.ga_branches, dtype=np.uint8)

        self.input_size = (
            24
            + self.ghr_size
            + 24
            + sum(self.lhr_configs[i][0] * input_params[i + 2] for i in range(3))
            + len(self.ga)
        )

    def extract_features(self, pc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        #pc_bits_repeated = np.tile(pc_bits, self.pc_times)

        # LHRs
        lhr_features = []
        #lhr_times_list = [
        #    self.lhr1_times,
        #    self.lhr2_times,
        #    self.lhr3_times,
        #]
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            #if lhr_times_list[i] > 0:
                index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
                lhr = self.lhrs[i][index]
                #lhr_repeated = np.tile(lhr, lhr_times_list[i])
                lhr_features.append(lhr)
        lhr_features_combined = (
            np.concatenate(lhr_features) if lhr_features else np.array([], dtype=np.uint8)
        )

        #ghr_repeated = np.tile(self.ghr, self.ghr_times)
        effective_xor_len = min(self.ghr_size, len(pc_bits))
        pc_bits_for_xor = pc_bits[-effective_xor_len:]
        ghr_for_xor = self.ghr[-effective_xor_len:]
        pc_ghr_xor = np.bitwise_xor(pc_bits_for_xor, ghr_for_xor)
        #pc_ghr_xor_repeated = np.tile(pc_ghr_xor, self.pc_ghr_times)
        
        #ga_repeated = (
        #    np.tile(self.ga, self.ga_times) if self.ga_times > 0 else np.array([], dtype=np.uint8)
        #)
        #ghr_ga_features = np.concatenate([self.ghr, self.ga])

        return pc_bits, pc_ghr_xor, lhr_features_combined, self.ghr, self.ga

    def get_input_pieces(
        self,
        pc_features: np.ndarray,
        xor_features: np.ndarray,
        lhr_features: np.ndarray,
        ga_features: np.ndarray,
        ghr_features: np.ndarray,
    ) -> Tuple[List[bytes], List[bytes], List[bytes]]:  # Retorna tupla de listas
        pc_pieces = self._get_pieces(pc_features, self.num_pc_filters, self.seed)
        lhr_pieces = self._get_pieces(lhr_features, self.num_lhr_filters, self.seed)
        xor_pieces = self._get_pieces(xor_features, self.num_xor_filters, self.seed)
        ghr_pieces = self._get_pieces(ghr_features, self.num_ghr_filters, self.seed) 
        ga_pieces = self._get_pieces(ga_features, self.num_ga_filters, self.seed) 
        return pc_pieces, xor_pieces, lhr_pieces, ghr_pieces, ga_pieces

    def _get_pieces(self, features: np.ndarray, num_filters: int, seed: int) -> List[bytes]:
        binary_input = "".join(list(map(str, features.tolist())))
        #indices = list(range(len(binary_input)))
        #random.seed(seed)
        #random.shuffle(indices)
        #shuffled_binary = "".join(binary_input[i] for i in indices)
        chunk_size = len(binary_input) // num_filters
        chunks = [
            binary_input[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_filters)
        ]
        remainder = len(binary_input) % num_filters
        for i in range(remainder):
            chunks[i] += binary_input[num_filters * chunk_size + i]
        return [chunk.encode() for chunk in chunks]

    def predict_and_train(self, pc: int, outcome: int):
        pc_features, xor_features, lhr_features, ghr_features, ga_features = self.extract_features(pc)
        pc_pieces, xor_pieces, lhr_pieces, ghr_pieces,ga_pieces = self.get_input_pieces(
            pc_features, xor_features, lhr_features, ga_features,ghr_features
        )

        pc_count_0 = self.pc_discriminators[0].get_count(pc_pieces)
        pc_count_1 = self.pc_discriminators[1].get_count(pc_pieces)

        lhr_count_0 = self.lhr_discriminators[0].get_count(lhr_pieces)
        lhr_count_1 = self.lhr_discriminators[1].get_count(lhr_pieces)

        ghr_count_0 = self.ghr_discriminators[0].get_count(ghr_pieces)
        ghr_count_1 = self.ghr_discriminators[1].get_count(ghr_pieces)

        ga_count_0 = self.ga_discriminators[0].get_count(ga_pieces)
        ga_count_1 = self.ga_discriminators[1].get_count(ga_pieces)

        xor_count_0 = self.xor_discriminators[0].get_count(xor_pieces)
        xor_count_1 = self.xor_discriminators[1].get_count(xor_pieces)

        individual_predictions = {
            'pc': (pc_count_0, pc_count_1),
            'lhr': (lhr_count_0, lhr_count_1),
            'ghr': (ghr_count_0, ghr_count_1),
            'ga': (ga_count_0, ga_count_1),
            'xor': (xor_count_0, xor_count_1)
        }

        prediction = self._tournament_predict(
            pc_count_0, 
            pc_count_1, 
            xor_count_0, 
            xor_count_1, 
            lhr_count_0, 
            lhr_count_1, 
            ghr_count_0, 
            ghr_count_1,
            ga_count_0, 
            ga_count_1
        )

        if prediction != outcome:
            self.pc_discriminators[outcome].train(pc_pieces)
            self.lhr_discriminators[outcome].train(lhr_pieces)
            self.ghr_discriminators[outcome].train(ghr_pieces)
            self.ga_discriminators[outcome].train(ga_pieces)
            self.xor_discriminators[outcome].train(xor_pieces)

            self.pc_discriminators[prediction].forget(pc_pieces)
            self.lhr_discriminators[prediction].forget(lhr_pieces)
            self.ghr_discriminators[prediction].forget(ghr_pieces)
            self.ga_discriminators[prediction].forget(ga_pieces)
            self.xor_discriminators[prediction].forget(xor_pieces)

        self.update_dynamic_weights(individual_predictions, outcome)

        self._update_histories(pc, outcome)
        return prediction == outcome

    def _tournament_predict(
        self,
        pc_count_0: int,
        pc_count_1: int,
        xor_count_0: int,
        xor_count_1: int,
        lhr_count_0: int,
        lhr_count_1: int,
        ghr_count_0: int,
        ghr_count_1: int,
        ga_count_0: int,
        ga_count_1: int,
    ) -> int:
        overall_count_0 = (self.dynamic_weights['pc'] * pc_count_0 + 
                          self.dynamic_weights['lhr'] * lhr_count_0 + 
                          self.dynamic_weights['ghr'] * ghr_count_0 + 
                          self.dynamic_weights['ga'] * ga_count_0 + 
                          self.dynamic_weights['xor'] * xor_count_0)
        
        overall_count_1 = (self.dynamic_weights['pc'] * pc_count_1 + 
                          self.dynamic_weights['lhr'] * lhr_count_1 + 
                          self.dynamic_weights['ghr'] * ghr_count_1 + 
                          self.dynamic_weights['ga'] * ga_count_1 + 
                          self.dynamic_weights['xor'] * xor_count_1)

        return 0 if overall_count_0 > overall_count_1 else 1

    def apply_bleaching(self):
        for disc in self.pc_discriminators:
            disc.binarize(self.pc_bleaching_threshold)

        for disc in self.lhr_discriminators:
            disc.binarize(self.lhr_bleaching_threshold)

        for disc in self.ghr_discriminators:
            disc.binarize(self.ghr_bleaching_threshold)

        for disc in self.ga_discriminators:
            disc.binarize(self.ga_bleaching_threshold)

        for disc in self.xor_discriminators:
            disc.binarize(self.xor_bleaching_threshold)

    def update_dynamic_weights(self, predictions: dict, outcome: int):
        for discriminator_name, (count_0, count_1) in predictions.items():
            discriminator_prediction = 0 if count_0 > count_1 else 1
            
            if discriminator_prediction == outcome:
                self.dynamic_weights[discriminator_name] += self.weight_adjustment_rate
            else:
                self.dynamic_weights[discriminator_name] -= self.weight_adjustment_rate
            
            self.dynamic_weights[discriminator_name] = max(
                self.min_weight, 
                min(self.max_weight, self.dynamic_weights[discriminator_name])
            )

    def get_dynamic_weights(self):
        return self.dynamic_weights.copy()

    def reset_dynamic_weights(self):
        self.dynamic_weights = {
            'pc': 1.0,
            'lhr': 1.0,
            'ghr': 1.0,
            'ga': 1.0,
            'xor': 1.0
        }

    def _update_histories(self, pc: int, outcome: int):
        self.ghr = np.roll(self.ghr, -1)
        self.ghr[-1] = outcome

        pc_bits = np.array(
            [int(b) for b in format(pc & ((1 << 24) - 1), "024b")], dtype=np.uint8
        )
        for i, (length, bits_pc) in enumerate(self.lhr_configs):
            index = int("".join(map(str, pc_bits[-bits_pc:])), 2)
            self.lhrs[i][index] = np.roll(self.lhrs[i][index], -1)
            self.lhrs[i][index][-1] = outcome

        new_bits = pc_bits[-self.ga_lower :]
        self.ga = np.roll(self.ga, -self.ga_lower)
        self.ga[-self.ga_lower :] = new_bits