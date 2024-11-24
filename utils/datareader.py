# from https://github.com/NiceAesth/maniera/blob/master/maniera/calculator.py

import math

class DataReader:
    __slots__ = ('osupath', 'mods', 'score', 'od', 'keys', 'notes', 'pp', 'sr')

    def __init__(self, osupath: str) -> None: #, mods: int, score: int) -> None:
        """Initialize Maniera calculator."""
        self.osupath = osupath
        #self.mods = mods
        #self.score = score
        self.od = 0
        self.keys = 0
        self.notes = []
        self.pp = 0.0
        self.sr = 0.0

        self.__parseBeatmapFile()

    def __parseNote(self, line: str) -> dict[str, object]:
        """Parse a note text into a note dict. (Internal use)."""
        m = line.split(',')
        if len(m) != 6:
            return

        x = float(m[0])
        start_t = float(m[2])
        end_t = float(m[5].split(':', 1)[0])
        if not end_t:
            end_t = start_t

        return {
            'key': math.floor( x * self.keys / 512 ),
            'start_t': start_t,
            'end_t': end_t,
            'overall_strain': 1,
            'individual_strain': [0] * self.keys
        }

    def __parseBeatmapFile(self) -> None:
        """Parse a beatmap file and set class variables. (Internal use)."""
        with open(self.osupath,  encoding='utf-8') as bmap:
            textContent = bmap.read()
            lines = textContent.splitlines()

        section_name = ""

        for line in lines:
            if not line or line[:2] == "//":
                continue

            if line[0] == "[" and line[-1] == "]":
                section_name = line[1:-1]
                continue

            if section_name == "General":
                key, val = line.split(':', maxsplit=1)
                if key == 'Mode' and val.lstrip() != '3':
                    raise RuntimeError('Maniera does not converted maps.')

            elif section_name == "Difficulty":
                key, val = line.split(':', maxsplit=1)
                if key == 'CircleSize':
                    self.keys = int(val.lstrip())
                elif key == 'OverallDifficulty':
                    self.od = float(val.lstrip())

            elif section_name == "HitObjects":
                note = self.__parseNote(line)
                if note:
                    self.notes.append(note)

        self.notes.sort(key=lambda note: note['start_t'])
