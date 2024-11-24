# from https://github.com/NiceAesth/maniera/blob/master/maniera/calculator.py
if __name__=="__main__":
    import datareader
else:
    from utils import datareader

import math

class Calculate():
    def _calculateStars(self) -> float:
        """Calculate star rating. (Internal use)."""
        # NOTE: make sure this is called before _calculatePP

        if not self.notes:
            return

        if self.mods & 64: # DT
            time_scale = 1.5
        elif self.mods & 256: # HT
            time_scale = 0.75
        else:
            time_scale = 1.0

        # Constants
        strain_step = 400 * time_scale
        weight_decay_base = 0.9
        individual_decay_base = 0.125
        overall_decay_base = 0.3
        star_scaling_factor = 0.018

        # Get strain of each note
        held_until = [0] * self.keys
        previous_note = self.notes[0]

        for note in self.notes[1:]:
            time_elapsed = (note['start_t'] - previous_note['start_t']) / time_scale / 1000
            individual_decay = individual_decay_base ** time_elapsed
            overall_decay = overall_decay_base ** time_elapsed
            hold_factor = 1
            hold_addition = 0

            for i in range(self.keys):
                if note['start_t'] < held_until[i] and note['end_t'] > held_until[i]:
                    hold_addition = 1
                elif note['end_t'] == held_until[i]:
                    hold_addition = 0
                elif note['end_t'] < held_until[i]:
                    hold_factor = 1.25
                note['individual_strain'][i] = previous_note['individual_strain'][i] * individual_decay

            held_until[note['key']] = note['end_t']

            note['individual_strain'][note['key']] += 2 * hold_factor
            note['overall_strain'] = previous_note['overall_strain'] * overall_decay + (1 + hold_addition) * hold_factor

            previous_note = note

        # Get difficulty for each interval
        strain_table = []
        max_strain = 0
        interval_end_time = strain_step
        previous_note = None

        for note in self.notes:
            while note['start_t'] > interval_end_time:
                strain_table.append(max_strain)

                if not previous_note:
                    max_strain = 0
                else:
                    individual_decay = individual_decay_base ** ( (interval_end_time - previous_note['start_t']) / 1000)
                    overall_decay = overall_decay_base ** ( (interval_end_time - previous_note['start_t']) / 1000)
                    max_strain = previous_note['individual_strain'][previous_note['key']] * individual_decay + previous_note['overall_strain'] * overall_decay

                interval_end_time += strain_step

            strain = note['individual_strain'][note['key']] + note['overall_strain']
            if strain > max_strain:
                max_strain = strain
            previous_note = note

        # Get total difficulty
        difficulty = 0
        weight = 1
        strain_table.sort(reverse=True)
        for i in strain_table:
            difficulty += i * weight
            weight *= weight_decay_base

        return difficulty * star_scaling_factor

    def _calculatePP(self) -> float:
        """Calculate PP. To be run only after _calculateStars. (Internal use)."""
        score_rate = 1.0

        if self.mods & 1: # NF
            score_rate *= 0.5
        if self.mods & 2: # EZ
            score_rate *= 0.5
        if self.mods & 256: # HT
            score_rate *= 0.5

        real_score = self.score / score_rate

        hit300_window = 34 + 3 * ( min( 10, max( 0, 10 - self.od ) ) )
        strain_value = (5 * max(1, self.sr / 0.2) - 4) ** 2.2 / 135 * (1 + 0.1 * min(1, len(self.notes) / 1500))

        if real_score <= 500000:
            strain_value = 0
        elif real_score <= 600000:
            strain_value *= ((real_score - 500000) / 100000 * 0.3)
        elif real_score <= 700000:
            strain_value *= (0.3 + (real_score - 600000) / 100000 * 0.25)
        elif real_score <= 800000:
            strain_value *= (0.55 + (real_score - 700000) / 100000 * 0.20)
        elif real_score <= 900000:
            strain_value *= (0.75 + (real_score - 800000) / 100000 * 0.15)
        else:
            strain_value *= (0.9 + (real_score - 900000) / 100000 * 0.1)

        acc_value = max(0, 0.2 - ( (hit300_window - 34) * 0.006667 ) ) * strain_value * ( max(0, real_score - 960000) / 40000) ** 1.1

        pp_multiplier = 0.8
        if self.mods & 1: # NF
            pp_multiplier *= 0.9
        if self.mods & 2: # EZ
            pp_multiplier *= 0.5

        return (strain_value ** 1.1 + acc_value ** 1.1) ** (1 / 1.1) * pp_multiplier

    def calculate(self) -> None:
        """Calculates PP and star rating."""
        self.sr = self._calculateStars()
        self.pp = self._calculatePP()