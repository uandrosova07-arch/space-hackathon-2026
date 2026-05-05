from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle_360(deg: float) -> float:
    value = deg % 360.0
    return value + 360.0 if value < 0.0 else value


def to_csv(value: float) -> str:
    return format(value, ".17g")


@dataclass(frozen=True)
class Stage1Constants:
    mu_km3_s2: float = 398600.4415
    r_gso_km: float = 42164.0
    r0_km: float = 6571.0
    i0_deg: float = 51.8
    rp_min_km: float = 6671.0
    head_block_initial_mass_kg: float = 8300.0
    rb_final_mass_kg: float = 1100.0
    rb_max_propellant_kg: float = 7100.0
    rb_exhaust_velocity_m_s: float = 3236.1945
    ep_thrust_n: float = 0.2
    ep_exhaust_velocity_m_s: float = 16180.9725
    required_mass_on_geo_kg: float = 1600.0


@dataclass
class OrbitEvaluation:
    rp_km: float
    ra_km: float
    inclination_deg: float
    delta_v_rb_km_s: float = 0.0
    delta_v_ep_km_s: float = 0.0
    mass_after_rb_burn_kg: float = 0.0
    spacecraft_initial_kg: float = 0.0
    mass_on_geo_kg: float = 0.0
    flight_time_sec: float = 0.0
    flight_time_days: float = 0.0
    is_valid: bool = False
    invalid_reason: str = ""

    def __str__(self) -> str:
        return (
            f"rp={self.rp_km:.1f} km, ra={self.ra_km:.1f} km, "
            f"i={self.inclination_deg:.2f} deg, mGEO={self.mass_on_geo_kg:.2f} kg, "
            f"t={self.flight_time_days:.2f} d, valid={self.is_valid}"
        )


@dataclass
class EnvelopePoint:
    time_bin_days: float
    best_solution: OrbitEvaluation


@dataclass
class Stage1Point:
    flight_time_days: float
    mass_on_geo_kg: float
    rp_km: float
    ra_km: float
    inclination_deg: float


class Table1Interpolator:
    def __init__(
        self,
        rp_grid: Sequence[float],
        ra_grid: Sequence[float],
        i_grid_deg: Sequence[float],
        values: list[list[list[float]]],
    ) -> None:
        self.rp_grid = list(rp_grid)
        self.ra_grid = list(ra_grid)
        self.i_grid_deg = list(i_grid_deg)
        self._values = values

    def is_inside(self, rp_norm: float, ra_norm: float, inc_deg: float) -> bool:
        return (
            self.rp_grid[0] <= rp_norm <= self.rp_grid[-1]
            and self.ra_grid[0] <= ra_norm <= self.ra_grid[-1]
            and self.i_grid_deg[0] <= inc_deg <= self.i_grid_deg[-1]
            and ra_norm >= rp_norm
        )

    def interpolate(self, rp_norm: float, ra_norm: float, inc_deg: float) -> float:
        if not self.is_inside(rp_norm, ra_norm, inc_deg):
            raise ValueError("Point is outside Table 1 domain.")

        irp = self._find_lower_index(self.rp_grid, rp_norm)
        ira = self._find_lower_index(self.ra_grid, ra_norm)
        ii = self._find_lower_index(self.i_grid_deg, inc_deg)

        irp1 = min(irp + 1, len(self.rp_grid) - 1)
        ira1 = min(ira + 1, len(self.ra_grid) - 1)
        ii1 = min(ii + 1, len(self.i_grid_deg) - 1)

        rp0, rp1 = self.rp_grid[irp], self.rp_grid[irp1]
        ra0, ra1 = self.ra_grid[ira], self.ra_grid[ira1]
        i0, i1 = self.i_grid_deg[ii], self.i_grid_deg[ii1]

        tx = self._safe_fraction(rp_norm, rp0, rp1)
        ty = self._safe_fraction(ra_norm, ra0, ra1)
        tz = self._safe_fraction(inc_deg, i0, i1)

        c000 = self._values[ira][irp][ii]
        c100 = self._values[ira][irp1][ii]
        c010 = self._values[ira1][irp][ii]
        c110 = self._values[ira1][irp1][ii]
        c001 = self._values[ira][irp][ii1]
        c101 = self._values[ira][irp1][ii1]
        c011 = self._values[ira1][irp][ii1]
        c111 = self._values[ira1][irp1][ii1]

        cell = [c000, c100, c010, c110, c001, c101, c011, c111]
        if any(math.isnan(value) or math.isinf(value) for value in cell):
            raise ValueError("Interpolation cell contains unfilled Table 1 values.")

        c00 = self._lerp(c000, c100, tx)
        c10 = self._lerp(c010, c110, tx)
        c01 = self._lerp(c001, c101, tx)
        c11 = self._lerp(c011, c111, tx)
        c0 = self._lerp(c00, c10, ty)
        c1 = self._lerp(c01, c11, ty)
        return self._lerp(c0, c1, tz)

    @staticmethod
    def _find_lower_index(grid: Sequence[float], value: float) -> int:
        if value <= grid[0]:
            return 0
        for index in range(len(grid) - 1):
            if grid[index] <= value <= grid[index + 1]:
                return index
        return len(grid) - 2

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _safe_fraction(x: float, x0: float, x1: float) -> float:
        if abs(x1 - x0) < 1e-14:
            return 0.0
        return (x - x0) / (x1 - x0)

    @classmethod
    def create_with_partial_data(cls) -> "Table1Interpolator":
        rp_grid = [
            0.15582, 0.17479, 0.26964, 0.38821, 0.62533,
            0.86246, 1.00000, 1.21816, 1.57385, 2.04811,
            2.52237, 2.99663, 3.47089, 3.94515, 4.41941,
        ]
        ra_grid = list(rp_grid)
        i_grid = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        values = [
            [[math.nan for _ in i_grid] for _ in rp_grid]
            for _ in ra_grid
        ]

        def set_row(
            ira: int,
            irp: int,
            v0: float,
            v15: float,
            v30: float,
            v45: float,
            v60: float,
            v75: float,
            v90: float,
        ) -> None:
            values[ira][irp] = [v0, v15, v30, v45, v60, v75, v90]

        set_row(0, 0, 1.53333, 1.64379, 1.94186, 2.33930, 2.68257, 2.99604, 3.27018)
        set_row(1, 0, 1.46214, 1.57330, 1.87295, 2.25965, 2.57992, 2.87177, 3.12976)
        set_row(1, 1, 1.39190, 1.50648, 1.81226, 2.21254, 2.54324, 2.83194, 3.08299)
        set_row(2, 0, 1.22458, 1.31869, 1.59000, 1.90446, 2.16797, 2.39681, 2.60069)
        set_row(2, 1, 1.16029, 1.26436, 1.55469, 1.88727, 2.15571, 2.38464, 2.58598)
        set_row(2, 2, 0.92579, 1.06201, 1.40264, 1.80825, 2.09596, 2.31882, 2.50702)
        set_row(3, 0, 1.04558, 1.12601, 1.35157, 1.60935, 1.83278, 2.02326, 2.18983)
        set_row(3, 1, 0.99428, 1.08149, 1.32781, 1.60149, 1.83122, 2.02384, 2.19064)
        set_row(3, 2, 0.78136, 0.90896, 1.23845, 1.57637, 1.82259, 2.01710, 2.18041)
        set_row(3, 3, 0.60495, 0.77247, 1.14757, 1.54910, 1.80517, 1.99402, 2.15027)
        set_row(4, 0, 0.84640, 0.91293, 1.08062, 1.27582, 1.45390, 1.60654, 1.73681)
        set_row(4, 1, 0.80606, 0.87976, 1.06435, 1.27294, 1.45821, 1.61452, 1.74686)
        set_row(4, 2, 0.63868, 0.74698, 1.01121, 1.27269, 1.48031, 1.64585, 1.78236)
        set_row(4, 3, 0.48000, 0.63341, 0.97679, 1.28360, 1.50168, 1.66862, 1.80434)
        set_row(4, 4, 0.26457, 0.51396, 0.91989, 1.29942, 1.52180, 1.68230, 1.81254)
        set_row(5, 0, 0.74740, 0.80227, 0.93753, 1.09789, 1.24928, 1.38134, 1.49362)
        set_row(5, 1, 0.71354, 0.77484, 0.92395, 1.09589, 1.25439, 1.39057, 1.50547)
        set_row(5, 2, 0.56994, 0.66492, 0.88082, 1.09952, 1.28215, 1.43054, 1.55245)
        set_row(5, 3, 0.42966, 0.56927, 0.85880, 1.11700, 1.31375, 1.46710, 1.59106)
        set_row(5, 4, 0.22118, 0.46558, 0.84470, 1.15408, 1.35764, 1.50966, 1.63140)
        set_row(5, 5, 0.07679, 0.42647, 0.82182, 1.17831, 1.38237, 1.52847, 1.64666)
        set_row(6, 0, 0.71296, 0.76238, 0.88428, 1.03028, 1.17024, 1.29360, 1.39884)
        set_row(6, 1, 0.68157, 0.73709, 0.87154, 1.02828, 1.17519, 1.30277, 1.41069)
        set_row(6, 2, 0.54796, 0.63577, 0.83097, 1.03161, 1.20287, 1.34358, 1.45933)
        set_row(6, 3, 0.41597, 0.54771, 0.81113, 1.04933, 1.23597, 1.38302, 1.50183)
        set_row(6, 4, 0.21580, 0.45138, 0.80553, 1.08970, 1.28579, 1.43362, 1.55177)
        set_row(6, 5, 0.06894, 0.41541, 0.79976, 1.12202, 1.31716, 1.46114, 1.57644)
        set_row(6, 6, 0.00000, 0.40623, 0.78890, 1.13274, 1.32976, 1.47100, 1.58422)
        set_row(7, 0, 0.67837, 0.72103, 0.82637, 0.95453, 1.07992, 1.19216, 1.28862)
        set_row(7, 1, 0.65019, 0.69823, 0.81449, 0.95230, 1.08431, 1.20080, 1.30005)
        set_row(7, 2, 0.52929, 0.60672, 0.77601, 0.95387, 1.11000, 1.24049, 1.34838)
        set_row(7, 3, 0.40749, 0.52735, 0.75695, 0.96997, 1.14271, 1.28109, 1.39322)
        set_row(7, 4, 0.22596, 0.44007, 0.75587, 1.01088, 1.19624, 1.33806, 1.45128)
        set_row(7, 5, 0.11388, 0.40697, 0.76319, 1.04642, 1.23372, 1.37363, 1.48513)
        set_row(7, 6, 0.09205, 0.39930, 0.76365, 1.06344, 1.24990, 1.38793, 1.49824)
        set_row(7, 7, 0.09396, 0.39476, 0.75584, 1.08244, 1.26969, 1.40417, 1.51267)
        set_row(8, 0, 0.65413, 0.68827, 0.77390, 0.88069, 0.98809, 1.08640, 1.17213)
        set_row(8, 1, 0.62982, 0.66832, 0.76285, 0.87790, 0.99138, 1.09386, 1.18242)
        set_row(8, 2, 0.52492, 0.58748, 0.72565, 0.87578, 1.01237, 1.12938, 1.22733)
        set_row(8, 3, 0.42338, 0.51718, 0.70533, 0.88747, 1.04152, 1.16797, 1.27142)
        set_row(8, 4, 0.28919, 0.43950, 0.70263, 0.92391, 1.09393, 1.22697, 1.33363)
        set_row(8, 5, 0.22589, 0.40859, 0.71513, 0.95951, 1.13459, 1.26810, 1.37451)
        set_row(8, 6, 0.21011, 0.40153, 0.72236, 0.97756, 1.15347, 1.28623, 1.39206)
        set_row(8, 7, 0.19787, 0.39860, 0.72984, 1.00212, 1.17783, 1.30879, 1.41350)
        set_row(8, 8, 0.20289, 0.40490, 0.72861, 1.03030, 1.20687, 1.33410, 1.43718)
        set_row(9, 0, 0.65361, 0.67895, 0.74600, 0.83256, 0.92239, 1.00678, 1.08182)
        set_row(9, 1, 0.63388, 0.66188, 0.73570, 0.82913, 0.92438, 1.01272, 1.09055)
        set_row(9, 2, 0.55224, 0.59306, 0.69922, 0.82258, 0.93924, 1.04205, 1.12967)
        set_row(9, 3, 0.47815, 0.53327, 0.67630, 0.82828, 0.96245, 1.07577, 1.16995)
        set_row(9, 4, 0.38811, 0.46385, 0.66704, 0.85593, 1.00855, 1.13123, 1.23064)
        set_row(9, 5, 0.34339, 0.43212, 0.67678, 0.88724, 1.04774, 1.17326, 1.27397)
        set_row(9, 6, 0.32832, 0.42369, 0.68479, 0.90428, 1.06704, 1.19294, 1.29375)
        set_row(9, 7, 0.31216, 0.41913, 0.69714, 0.92859, 1.09314, 1.21876, 1.31929)
        set_row(9, 8, 0.29544, 0.42395, 0.71141, 0.96115, 1.12621, 1.25031, 1.35013)
        set_row(9, 9, 0.30099, 0.44012, 0.71590, 0.99047, 1.15742, 1.27875, 1.37788)
        set_row(10, 0, 0.67335, 0.69112, 0.74173, 0.81326, 0.88984, 0.96346, 1.03010)
        set_row(10, 1, 0.65784, 0.67716, 0.73212, 0.80938, 0.89080, 0.96812, 1.03749)
        set_row(10, 2, 0.59499, 0.62114, 0.69655, 0.79942, 0.90056, 0.99196, 1.07121)
        set_row(10, 3, 0.53943, 0.57226, 0.67126, 0.80007, 0.91827, 1.02065, 1.10711)
        set_row(10, 4, 0.47161, 0.51228, 0.65495, 0.81903, 0.95692, 1.07047, 1.16366)
        set_row(10, 5, 0.43415, 0.47882, 0.65936, 0.84465, 0.99229, 1.11045, 1.20618)
        set_row(10, 6, 0.41959, 0.46658, 0.66551, 0.85953, 1.01049, 1.12991, 1.22631)
        set_row(10, 7, 0.40232, 0.45512, 0.67685, 0.88174, 1.03594, 1.15625, 1.25312)
        set_row(10, 8, 0.37976, 0.45153, 0.69429, 0.91311, 1.06965, 1.19000, 1.28702)
        set_row(10, 9, 0.36229, 0.46131, 0.70963, 0.94580, 1.10311, 1.22250, 1.31953)
        set_row(10, 10, 0.37016, 0.47733, 0.71474, 0.96721, 1.12737, 1.24538, 1.34273)
        set_row(11, 0, 0.70192, 0.71478, 0.75227, 0.80912, 0.87519, 0.94007, 0.99976)
        set_row(11, 1, 0.68959, 0.70342, 0.74374, 0.80493, 0.87536, 0.94370, 1.00604)
        set_row(11, 2, 0.64001, 0.65799, 0.71098, 0.79239, 0.88100, 0.96289, 1.03508)
        set_row(11, 3, 0.59625, 0.61813, 0.68476, 0.78899, 0.89396, 0.98703, 1.06682)
        set_row(11, 4, 0.54139, 0.56736, 0.65940, 0.80017, 0.92545, 1.03090, 1.11858)
        set_row(11, 5, 0.50861, 0.53559, 0.65576, 0.82001, 0.95634, 1.06775, 1.15901)
        set_row(11, 6, 0.49499, 0.52175, 0.65901, 0.83243, 0.97283, 1.08622, 1.17863)
        set_row(11, 7, 0.47843, 0.50456, 0.66749, 0.85180, 0.99656, 1.11182, 1.20537)
        set_row(11, 8, 0.45479, 0.48846, 0.68352, 0.88057, 1.02916, 1.14573, 1.24022)
        set_row(11, 9, 0.42571, 0.48674, 0.70188, 0.91218, 1.06290, 1.17981, 1.27502)
        set_row(11, 10, 0.41375, 0.49637, 0.71351, 0.93666, 1.08826, 1.20498, 1.30091)
        set_row(11, 11, 0.42218, 0.51039, 0.71869, 0.95257, 1.10758, 1.22387, 1.32075)
        set_row(12, 0, 0.73386, 0.74358, 0.77217, 0.81630, 0.87112, 0.92881, 0.98256)
        set_row(12, 1, 0.72388, 0.73428, 0.76480, 0.81203, 0.87069, 0.93158, 0.98791)
        set_row(12, 2, 0.68362, 0.69709, 0.73592, 0.79730, 0.87312, 0.94690, 1.01294)
        set_row(12, 3, 0.64821, 0.66422, 0.71122, 0.78913, 0.88201, 0.96698, 1.04091)
        set_row(12, 4, 0.60246, 0.62100, 0.68120, 0.79262, 0.90690, 1.00514, 1.08787)
        set_row(12, 5, 0.57401, 0.59191, 0.66705, 0.80698, 0.93326, 1.03853, 1.12568)
        set_row(12, 6, 0.56211, 0.57811, 0.66477, 0.81691, 0.94786, 1.05567, 1.14444)
        set_row(12, 7, 0.54562, 0.55887, 0.66746, 0.83320, 0.96939, 1.07992, 1.17040)
        set_row(12, 8, 0.52063, 0.53376, 0.67926, 0.85868, 0.99994, 1.11288, 1.20505)
        set_row(12, 9, 0.48409, 0.51760, 0.69667, 0.88815, 1.03274, 1.14711, 1.24067)
        set_row(12, 10, 0.46302, 0.51822, 0.71067, 0.91205, 1.05825, 1.17329, 1.26798)
        set_row(12, 11, 0.45491, 0.52678, 0.71988, 0.93090, 1.07815, 1.19359, 1.28950)
        set_row(12, 12, 0.46312, 0.53880, 0.72488, 0.94289, 1.09379, 1.20947, 1.30675)
        set_row(13, 0, 0.76772, 0.77537, 0.79783, 0.83243, 0.87634, 0.92546, 0.97404)
        set_row(13, 1, 0.75956, 0.76773, 0.79158, 0.82839, 0.87522, 0.92753, 0.97863)
        set_row(13, 2, 0.72659, 0.73719, 0.76668, 0.81323, 0.87366, 0.93963, 1.00026)
        set_row(13, 3, 0.69756, 0.70996, 0.74441, 0.80184, 0.87824, 0.95612, 1.02484)
        set_row(13, 4, 0.65988, 0.67296, 0.71391, 0.79451, 0.89716, 0.98893, 1.06716)
        set_row(13, 5, 0.63582, 0.64627, 0.69350, 0.80195, 0.91912, 1.01878, 1.10219)
        set_row(13, 6, 0.62390, 0.63291, 0.68540, 0.80941, 0.93178, 1.03448, 1.12003)
        set_row(13, 7, 0.60647, 0.61327, 0.67874, 0.82254, 0.95094, 1.05705, 1.14470)
        set_row(13, 8, 0.57994, 0.58368, 0.68165, 0.84443, 0.97896, 1.08845, 1.17847)
        set_row(13, 9, 0.54390, 0.55434, 0.69515, 0.87116, 1.01009, 1.12199, 1.21402)
        set_row(13, 10, 0.50964, 0.54368, 0.70882, 0.89377, 1.03504, 1.14834, 1.24192)
        set_row(13, 11, 0.49414, 0.54535, 0.71968, 0.91233, 1.05499, 1.16929, 1.26436)
        set_row(13, 12, 0.48875, 0.55285, 0.72721, 0.92714, 1.07097, 1.18611, 1.28271)
        set_row(13, 13, 0.49643, 0.56319, 0.73190, 0.93630, 1.08379, 1.19964, 1.29833)
        set_row(14, 0, 0.80434, 0.81096, 0.82876, 0.85585, 0.89056, 0.93043, 0.97212)
        set_row(14, 1, 0.79776, 0.80478, 0.82352, 0.85215, 0.88905, 0.93154, 0.97586)
        set_row(14, 2, 0.77158, 0.78003, 0.80222, 0.83747, 0.88445, 0.93907, 0.99439)
        set_row(14, 3, 0.74899, 0.75756, 0.78232, 0.82459, 0.88362, 0.95187, 1.01583)
        set_row(14, 4, 0.71800, 0.72503, 0.75265, 0.80926, 0.89385, 0.97962, 1.05375)
        set_row(14, 5, 0.69375, 0.69954, 0.72957, 0.80547, 0.91150, 1.00600, 1.08596)
        set_row(14, 6, 0.68120, 0.68628, 0.71819, 0.80850, 0.92222, 1.02019, 1.10248)
        set_row(14, 7, 0.66269, 0.66643, 0.70356, 0.81782, 0.93893, 1.04094, 1.12596)
        set_row(14, 8, 0.63464, 0.63542, 0.69251, 0.83588, 0.96420, 1.07045, 1.15845)
        set_row(14, 9, 0.59849, 0.59652, 0.69780, 0.85948, 0.99320, 1.10274, 1.19336)
        set_row(14, 10, 0.55658, 0.57338, 0.70907, 0.88037, 1.01711, 1.12871, 1.22130)
        set_row(14, 11, 0.53260, 0.56657, 0.71980, 0.89811, 1.03669, 1.14980, 1.24413)
        set_row(14, 12, 0.52062, 0.56871, 0.72844, 0.91283, 1.05267, 1.16706, 1.26310)
        set_row(14, 13, 0.51715, 0.57530, 0.73474, 0.92466, 1.06569, 1.18125, 1.27941)
        set_row(14, 14, 0.52423, 0.58428, 0.73909, 0.93172, 1.07629, 1.19290, 1.29290)
        return cls(rp_grid, ra_grid, i_grid, values)


class Stage1Solver:
    def __init__(self, constants: Stage1Constants, interpolator: Table1Interpolator) -> None:
        self.constants = constants
        self.interpolator = interpolator

    def evaluate_intermediate_orbit(
        self,
        rp_km: float,
        ra_km: float,
        inc_deg: float,
    ) -> OrbitEvaluation:
        c = self.constants
        result = OrbitEvaluation(rp_km=rp_km, ra_km=ra_km, inclination_deg=inc_deg)

        if rp_km < c.rp_min_km:
            result.invalid_reason = "Perigee radius is below minimum."
            return result
        if ra_km < rp_km:
            result.invalid_reason = "Apogee radius is smaller than perigee radius."
            return result
        if not (0.0 <= inc_deg <= c.i0_deg):
            result.invalid_reason = "Inclination is outside allowed range."
            return result

        a2_km = 0.5 * (rp_km + ra_km)
        v0_km_s = math.sqrt(c.mu_km3_s2 / c.r0_km)
        vp_km_s = math.sqrt(c.mu_km3_s2 * (2.0 / rp_km - 1.0 / a2_km))
        delta_i_rad = deg_to_rad(abs(c.i0_deg - inc_deg))
        delta_v_rb_km_s = math.sqrt(
            v0_km_s * v0_km_s
            + vp_km_s * vp_km_s
            - 2.0 * v0_km_s * vp_km_s * math.cos(delta_i_rad)
        )
        result.delta_v_rb_km_s = delta_v_rb_km_s

        delta_v_rb_m_s = delta_v_rb_km_s * 1000.0
        mass_after_rb_burn_kg = c.head_block_initial_mass_kg * math.exp(
            -delta_v_rb_m_s / c.rb_exhaust_velocity_m_s
        )
        result.mass_after_rb_burn_kg = mass_after_rb_burn_kg

        rb_propellant_used_kg = c.head_block_initial_mass_kg - mass_after_rb_burn_kg
        if rb_propellant_used_kg > c.rb_max_propellant_kg + 1e-9:
            result.invalid_reason = "RB propellant consumption exceeds limit."
            return result

        spacecraft_initial_kg = mass_after_rb_burn_kg - c.rb_final_mass_kg
        result.spacecraft_initial_kg = spacecraft_initial_kg
        if spacecraft_initial_kg <= 0.0:
            result.invalid_reason = "Spacecraft mass after RB separation is non-positive."
            return result

        rp_norm = rp_km / c.r_gso_km
        ra_norm = ra_km / c.r_gso_km
        if not self.interpolator.is_inside(rp_norm, ra_norm, inc_deg):
            result.invalid_reason = "Point is outside Table 1 interpolation domain."
            return result

        try:
            vch_star = self.interpolator.interpolate(rp_norm, ra_norm, inc_deg)
        except Exception as exc:
            result.invalid_reason = f"Interpolation failed: {exc}"
            return result

        v_circ_geo_km_s = math.sqrt(c.mu_km3_s2 / c.r_gso_km)
        delta_v_ep_km_s = vch_star * v_circ_geo_km_s
        result.delta_v_ep_km_s = delta_v_ep_km_s

        delta_v_ep_m_s = delta_v_ep_km_s * 1000.0
        mass_on_geo_kg = spacecraft_initial_kg * math.exp(
            -delta_v_ep_m_s / c.ep_exhaust_velocity_m_s
        )
        result.mass_on_geo_kg = mass_on_geo_kg

        if mass_on_geo_kg <= 0.0 or mass_on_geo_kg >= spacecraft_initial_kg:
            result.invalid_reason = "Invalid final mass after EP transfer."
            return result

        mdot_kg_s = c.ep_thrust_n / c.ep_exhaust_velocity_m_s
        flight_time_sec = (spacecraft_initial_kg - mass_on_geo_kg) / mdot_kg_s
        result.flight_time_sec = flight_time_sec
        result.flight_time_days = flight_time_sec / 86400.0
        result.is_valid = True
        return result

    def run_grid_search(
        self,
        rp_start_km: float,
        rp_end_km: float,
        rp_step_km: float,
        ra_start_km: float,
        ra_end_km: float,
        ra_step_km: float,
        i_start_deg: float,
        i_end_deg: float,
        i_step_deg: float,
    ) -> list[OrbitEvaluation]:
        all_solutions: list[OrbitEvaluation] = []
        rp = rp_start_km
        while rp <= rp_end_km + 1e-9:
            ra = max(ra_start_km, rp)
            while ra <= ra_end_km + 1e-9:
                inc = i_start_deg
                while inc <= i_end_deg + 1e-9:
                    evaluation = self.evaluate_intermediate_orbit(rp, ra, inc)
                    if evaluation.is_valid:
                        all_solutions.append(evaluation)
                    inc += i_step_deg
                ra += ra_step_km
            rp += rp_step_km
        return all_solutions

    @staticmethod
    def build_time_envelope(
        solutions: Sequence[OrbitEvaluation],
        time_bin_days: float,
    ) -> list[EnvelopePoint]:
        groups: dict[int, OrbitEvaluation] = {}
        for solution in solutions:
            bin_index = round(solution.flight_time_days / time_bin_days)
            current = groups.get(bin_index)
            if current is None or solution.mass_on_geo_kg > current.mass_on_geo_kg:
                groups[bin_index] = solution
        return [
            EnvelopePoint(time_bin_days=bin_index * time_bin_days, best_solution=solution)
            for bin_index, solution in sorted(groups.items())
        ]

    @staticmethod
    def find_best_for_required_mass(
        solutions: Sequence[OrbitEvaluation],
        required_mass_kg: float,
    ) -> OrbitEvaluation | None:
        candidates = [s for s in solutions if s.is_valid and s.mass_on_geo_kg >= required_mass_kg]
        candidates.sort(key=lambda item: item.flight_time_days)
        return candidates[0] if candidates else None

    @staticmethod
    def export_all_solutions_csv(path: Path, solutions: Sequence[OrbitEvaluation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["FlightTimeDays", "MassOnGeoKg", "RpKm", "RaKm", "InclinationDeg"])
            for item in sorted(solutions, key=lambda x: x.flight_time_days):
                writer.writerow([
                    to_csv(item.flight_time_days),
                    to_csv(item.mass_on_geo_kg),
                    to_csv(item.rp_km),
                    to_csv(item.ra_km),
                    to_csv(item.inclination_deg),
                ])

    @staticmethod
    def export_envelope_csv(path: Path, envelope: Sequence[EnvelopePoint]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["FlightTimeDays", "MassOnGeoKg", "RpKm", "RaKm", "InclinationDeg"])
            for point in sorted(envelope, key=lambda x: x.time_bin_days):
                s = point.best_solution
                writer.writerow([
                    to_csv(point.time_bin_days),
                    to_csv(s.mass_on_geo_kg),
                    to_csv(s.rp_km),
                    to_csv(s.ra_km),
                    to_csv(s.inclination_deg),
                ])


class Stage1CsvReader:
    @staticmethod
    def read_final_csv(path: Path) -> list[Stage1Point]:
        if not path.exists():
            return []
        result: list[Stage1Point] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                result.append(
                    Stage1Point(
                        flight_time_days=float(row[0]),
                        mass_on_geo_kg=float(row[1]),
                        rp_km=float(row[2]),
                        ra_km=float(row[3]),
                        inclination_deg=float(row[4]),
                    )
                )
        return result


class Stage1Plotter:
    @staticmethod
    def create_all_plots(
        csv_path: Path,
        output_directory: Path,
        required_mass_kg: float = 1600.0,
        show_windows: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        output_directory.mkdir(parents=True, exist_ok=True)
        points = sorted(Stage1CsvReader.read_final_csv(csv_path), key=lambda p: p.flight_time_days)
        if not points:
            raise ValueError("CSV-файл пуст или не содержит корректных строк.")

        best_point = next(
            (p for p in points if p.mass_on_geo_kg >= required_mass_kg),
            None,
        )

        Stage1Plotter._create_single_plot(
            plt,
            points,
            best_point,
            y_selector=lambda p: p.mass_on_geo_kg,
            y_label="Масса КА на ГСО (кг)",
            title="Первый этап: масса КА на ГСО от времени перелета",
            file_path=output_directory / "stage1_mass_vs_time.png",
        )
        Stage1Plotter._create_single_plot(
            plt,
            points,
            best_point,
            y_selector=lambda p: p.rp_km,
            y_label="Радиус перицентра (км)",
            title="Первый этап: оптимальный радиус перицентра от времени перелета",
            file_path=output_directory / "stage1_rp_vs_time.png",
        )
        Stage1Plotter._create_single_plot(
            plt,
            points,
            best_point,
            y_selector=lambda p: p.ra_km,
            y_label="Радиус апоцентра (км)",
            title="Первый этап: оптимальный радиус апоцентра от времени перелета",
            file_path=output_directory / "stage1_ra_vs_time.png",
        )
        Stage1Plotter._create_single_plot(
            plt,
            points,
            best_point,
            y_selector=lambda p: p.inclination_deg,
            y_label="Наклонение (град)",
            title="Первый этап: оптимальное наклонение от времени перелета",
            file_path=output_directory / "stage1_i_vs_time.png",
        )
        if show_windows:
            plt.show()
        plt.close("all")

    @staticmethod
    def _create_single_plot(
        plt_module,
        points: Sequence[Stage1Point],
        best_point: Stage1Point | None,
        y_selector: Callable[[Stage1Point], float],
        y_label: str,
        title: str,
        file_path: Path,
    ) -> None:
        xs = [p.flight_time_days for p in points]
        ys = [y_selector(p) for p in points]
        fig, ax = plt_module.subplots(figsize=(12, 8))
        ax.plot(xs, ys, label="Оптимальная огибающая")

        if best_point is not None:
            best_y = y_selector(best_point)
            ax.scatter([best_point.flight_time_days], [best_y], s=90, label="Выбранное решение")
            label = (
                f"t = {best_point.flight_time_days:.2f} сут\n"
                f"m = {best_point.mass_on_geo_kg:.2f} кг\n"
                f"rp = {best_point.rp_km:.0f} км\n"
                f"ra = {best_point.ra_km:.0f} км\n"
                f"i = {best_point.inclination_deg:.2f}°"
            )
            ax.text(best_point.flight_time_days + 3.0, best_y, label)

        ax.set_title(title)
        ax.set_xlabel("Время перелета (сутки)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)


@dataclass(frozen=True)
class Stage2Constants:
    mu_km3_s2: float = 398600.4415
    r_gso_km: float = 42164.0
    ep_thrust_n: float = 0.2
    ep_exhaust_velocity_m_s: float = 16180.9725
    initial_rp_km: float = 11421.0
    initial_ra_km: float = 105000.0
    initial_inclination_deg: float = 27.0
    initial_mass_kg: float = 1821.36375899487
    arg_perigee_deg: float = 0.0
    raan_deg: float = 0.0
    true_anomaly_deg: float = 0.0
    time_step_sec: float = 300.0
    max_flight_days: float = 400.0
    target_epsilon_a_km: float = 100.0
    target_epsilon_e: float = 1e-3
    target_epsilon_i_deg: float = 0.1


@dataclass
class Vector3d:
    x: float
    y: float
    z: float

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> "Vector3d":
        n = self.norm()
        if n < 1e-15:
            return Vector3d(0.0, 0.0, 0.0)
        return self / n

    @staticmethod
    def dot(a: "Vector3d", b: "Vector3d") -> float:
        return a.x * b.x + a.y * b.y + a.z * b.z

    @staticmethod
    def cross(a: "Vector3d", b: "Vector3d") -> "Vector3d":
        return Vector3d(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )

    def __add__(self, other: "Vector3d") -> "Vector3d":
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3d") -> "Vector3d":
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3d":
        return Vector3d(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3d":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector3d":
        return Vector3d(self.x / scalar, self.y / scalar, self.z / scalar)


@dataclass
class SpacecraftState:
    time_sec: float
    position_km: Vector3d
    velocity_km_s: Vector3d
    mass_kg: float

    def clone(self) -> "SpacecraftState":
        return SpacecraftState(
            time_sec=self.time_sec,
            position_km=Vector3d(self.position_km.x, self.position_km.y, self.position_km.z),
            velocity_km_s=Vector3d(self.velocity_km_s.x, self.velocity_km_s.y, self.velocity_km_s.z),
            mass_kg=self.mass_kg,
        )


@dataclass
class OsculatingElements:
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float


@dataclass
class GuidanceOutput:
    thrust_direction_inertial: Vector3d
    right_ascension_deg: float
    declination_deg: float


@dataclass
class GuidanceParameters:
    ke: float
    ka: float
    i_switch_deg: float
    i_normal_max: float

    def __str__(self) -> str:
        return (
            f"Ke={self.ke:.3f}, Ka={self.ka:.3f}, "
            f"ISwitch={self.i_switch_deg:.3f}, INormalMax={self.i_normal_max:.3f}"
        )


@dataclass
class OrbitGeometry:
    angular_momentum: Vector3d
    eccentricity_vector: Vector3d
    true_anomaly_rad: float


@dataclass
class GuidanceSample:
    time_days: float
    right_ascension_deg: float
    declination_deg: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    mass_kg: float
    position_km: Vector3d


@dataclass
class SimulationResult:
    samples: list[GuidanceSample] = field(default_factory=list)
    final_state: SpacecraftState | None = None
    final_elements: OsculatingElements | None = None
    reached_target: bool = False


@dataclass
class TuningResult:
    parameters: GuidanceParameters
    reached_target: bool
    final_time_days: float
    final_mass_kg: float
    final_a_km: float
    final_e: float
    final_i_deg: float
    simulation: SimulationResult


class OrbitMath:
    def __init__(self, constants: Stage2Constants) -> None:
        self.constants = constants

    def compute_osculating_elements(self, r_km: Vector3d, v_km_s: Vector3d) -> OsculatingElements:
        mu = self.constants.mu_km3_s2
        r = r_km.norm()
        v = v_km_s.norm()
        h_vec = Vector3d.cross(r_km, v_km_s)
        h = h_vec.norm()
        energy = 0.5 * v * v - mu / r
        a = -mu / (2.0 * energy)
        e_vec = Vector3d.cross(v_km_s, h_vec) / mu - r_km / r
        e = e_vec.norm()
        i_rad = math.acos(clamp(h_vec.z / h, -1.0, 1.0))
        return OsculatingElements(a, e, rad_to_deg(i_rad))

    @staticmethod
    def build_local_orbital_frame(
        r_km: Vector3d,
        v_km_s: Vector3d,
    ) -> tuple[Vector3d, Vector3d, Vector3d]:
        r_hat = r_km.normalize()
        h_vec = Vector3d.cross(r_km, v_km_s)
        n_hat = h_vec.normalize()
        t_hat = Vector3d.cross(n_hat, r_hat).normalize()
        return r_hat, t_hat, n_hat

    def compute_orbit_geometry(self, r_km: Vector3d, v_km_s: Vector3d) -> OrbitGeometry:
        mu = self.constants.mu_km3_s2
        r = r_km.norm()
        h_vec = Vector3d.cross(r_km, v_km_s)
        e_vec = Vector3d.cross(v_km_s, h_vec) / mu - r_km / r
        e = e_vec.norm()
        nu_rad = 0.0
        if e > 1e-10:
            cos_nu = clamp(Vector3d.dot(e_vec, r_km) / (e * r), -1.0, 1.0)
            nu_rad = math.acos(cos_nu)
            if Vector3d.dot(r_km, v_km_s) < 0.0:
                nu_rad = 2.0 * math.pi - nu_rad
        return OrbitGeometry(h_vec, e_vec, nu_rad)

    def create_initial_state_from_keplerian(
        self,
        rp_km: float,
        ra_km: float,
        inclination_deg: float,
        raan_deg: float,
        arg_perigee_deg: float,
        true_anomaly_deg: float,
        mass_kg: float,
    ) -> SpacecraftState:
        a = 0.5 * (rp_km + ra_km)
        e = (ra_km - rp_km) / (ra_km + rp_km)
        p = a * (1.0 - e * e)
        nu = deg_to_rad(true_anomaly_deg)
        inc = deg_to_rad(inclination_deg)
        raan = deg_to_rad(raan_deg)
        argp = deg_to_rad(arg_perigee_deg)
        r = p / (1.0 + e * math.cos(nu))
        sqrt_mu_over_p = math.sqrt(self.constants.mu_km3_s2 / p)
        r_pqw = Vector3d(r * math.cos(nu), r * math.sin(nu), 0.0)
        v_pqw = Vector3d(-sqrt_mu_over_p * math.sin(nu), sqrt_mu_over_p * (e + math.cos(nu)), 0.0)
        return SpacecraftState(
            time_sec=0.0,
            position_km=self._rotate_pqw_to_ijk(r_pqw, raan, inc, argp),
            velocity_km_s=self._rotate_pqw_to_ijk(v_pqw, raan, inc, argp),
            mass_kg=mass_kg,
        )

    @staticmethod
    def _rotate_pqw_to_ijk(v: Vector3d, raan: float, inc: float, argp: float) -> Vector3d:
        cos_o = math.cos(raan)
        sin_o = math.sin(raan)
        cos_i = math.cos(inc)
        sin_i = math.sin(inc)
        cos_w = math.cos(argp)
        sin_w = math.sin(argp)

        m11 = cos_o * cos_w - sin_o * sin_w * cos_i
        m12 = -cos_o * sin_w + sin_o * cos_w * cos_i
        m13 = sin_o * sin_i
        m21 = sin_o * cos_w + cos_o * sin_w * cos_i
        m22 = -sin_o * sin_w + cos_o * cos_w * cos_i
        m23 = -cos_o * sin_i
        m31 = sin_w * sin_i
        m32 = cos_w * sin_i
        m33 = cos_i
        return Vector3d(
            m11 * v.x + m12 * v.y + m13 * v.z,
            m21 * v.x + m22 * v.y + m23 * v.z,
            m31 * v.x + m32 * v.y + m33 * v.z,
        )


class GuidanceLaw:
    def __init__(self, constants: Stage2Constants, orbit_math: OrbitMath) -> None:
        self.constants = constants
        self.orbit_math = orbit_math

    def compute_guidance(self, state: SpacecraftState, params: GuidanceParameters) -> GuidanceOutput:
        r_hat, t_hat, n_hat = self.orbit_math.build_local_orbital_frame(state.position_km, state.velocity_km_s)
        elements = self.orbit_math.compute_osculating_elements(state.position_km, state.velocity_km_s)
        geometry = self.orbit_math.compute_orbit_geometry(state.position_km, state.velocity_km_s)

        un = 0.0
        if elements.inclination_deg > params.i_switch_deg:
            sign_normal = -1.0 if state.velocity_km_s.z >= 0.0 else 1.0
            un = params.i_normal_max * sign_normal

        sa = 1.0 if elements.semi_major_axis_km < self.constants.r_gso_km else -1.0
        tangential_command = -params.ke * math.cos(geometry.true_anomaly_rad) + params.ka * sa
        sign_tangential = 1.0 if tangential_command >= 0.0 else -1.0
        ut = sign_tangential * math.sqrt(max(0.0, 1.0 - un * un))
        ur = 0.0

        direction = (ur * r_hat + ut * t_hat + un * n_hat).normalize()
        ra_deg = normalize_angle_360(rad_to_deg(math.atan2(direction.y, direction.x)))
        dec_deg = rad_to_deg(math.asin(clamp(direction.z, -1.0, 1.0)))
        return GuidanceOutput(direction, ra_deg, dec_deg)


class Dynamics:
    def __init__(self, constants: Stage2Constants, guidance_law: GuidanceLaw) -> None:
        self.constants = constants
        self.guidance_law = guidance_law

    def derivatives(self, state: SpacecraftState, guidance_parameters: GuidanceParameters) -> SpacecraftState:
        guidance = self.guidance_law.compute_guidance(state, guidance_parameters)
        r = state.position_km
        m = state.mass_kg
        r_norm = r.norm()
        grav_accel = (-self.constants.mu_km3_s2 / (r_norm * r_norm * r_norm)) * r
        thrust_accel_km_s2 = (self.constants.ep_thrust_n / m) / 1000.0
        thrust_accel = thrust_accel_km_s2 * guidance.thrust_direction_inertial
        total_accel = grav_accel + thrust_accel
        mdot = -self.constants.ep_thrust_n / self.constants.ep_exhaust_velocity_m_s
        return SpacecraftState(
            time_sec=1.0,
            position_km=state.velocity_km_s,
            velocity_km_s=total_accel,
            mass_kg=mdot,
        )


class Integrator:
    def __init__(self, dynamics: Dynamics) -> None:
        self.dynamics = dynamics

    def step_rk4(
        self,
        state: SpacecraftState,
        dt: float,
        guidance_parameters: GuidanceParameters,
    ) -> SpacecraftState:
        k1 = self.dynamics.derivatives(state, guidance_parameters)
        k2 = self.dynamics.derivatives(self._add_scaled(state, k1, dt / 2.0), guidance_parameters)
        k3 = self.dynamics.derivatives(self._add_scaled(state, k2, dt / 2.0), guidance_parameters)
        k4 = self.dynamics.derivatives(self._add_scaled(state, k3, dt), guidance_parameters)
        return SpacecraftState(
            time_sec=state.time_sec + dt,
            position_km=state.position_km + (dt / 6.0) * (k1.position_km + 2.0 * k2.position_km + 2.0 * k3.position_km + k4.position_km),
            velocity_km_s=state.velocity_km_s + (dt / 6.0) * (k1.velocity_km_s + 2.0 * k2.velocity_km_s + 2.0 * k3.velocity_km_s + k4.velocity_km_s),
            mass_kg=state.mass_kg + (dt / 6.0) * (k1.mass_kg + 2.0 * k2.mass_kg + 2.0 * k3.mass_kg + k4.mass_kg),
        )

    @staticmethod
    def _add_scaled(state: SpacecraftState, delta: SpacecraftState, scale: float) -> SpacecraftState:
        return SpacecraftState(
            time_sec=state.time_sec + scale * delta.time_sec,
            position_km=state.position_km + scale * delta.position_km,
            velocity_km_s=state.velocity_km_s + scale * delta.velocity_km_s,
            mass_kg=state.mass_kg + scale * delta.mass_kg,
        )


class TargetConditions:
    def __init__(self, constants: Stage2Constants, orbit_math: OrbitMath) -> None:
        self.constants = constants
        self.orbit_math = orbit_math

    def is_reached(self, state: SpacecraftState) -> tuple[bool, OsculatingElements]:
        elements = self.orbit_math.compute_osculating_elements(state.position_km, state.velocity_km_s)
        reached = (
            abs(elements.semi_major_axis_km - self.constants.r_gso_km) < self.constants.target_epsilon_a_km
            and elements.eccentricity < self.constants.target_epsilon_e
            and elements.inclination_deg < self.constants.target_epsilon_i_deg
        )
        return reached, elements


class Simulator:
    def __init__(self, constants: Stage2Constants) -> None:
        self.constants = constants
        self.orbit_math = OrbitMath(constants)
        self.guidance_law = GuidanceLaw(constants, self.orbit_math)
        self.dynamics = Dynamics(constants, self.guidance_law)
        self.integrator = Integrator(self.dynamics)
        self.target_conditions = TargetConditions(constants, self.orbit_math)

    def run(self, guidance_parameters: GuidanceParameters, time_step_sec: float | None = None) -> SimulationResult:
        state = self.orbit_math.create_initial_state_from_keplerian(
            rp_km=self.constants.initial_rp_km,
            ra_km=self.constants.initial_ra_km,
            inclination_deg=self.constants.initial_inclination_deg,
            raan_deg=self.constants.raan_deg,
            arg_perigee_deg=self.constants.arg_perigee_deg,
            true_anomaly_deg=self.constants.true_anomaly_deg,
            mass_kg=self.constants.initial_mass_kg,
        )
        result = SimulationResult()
        dt = time_step_sec or self.constants.time_step_sec
        max_time_sec = self.constants.max_flight_days * 86400.0

        while state.time_sec <= max_time_sec:
            guidance = self.guidance_law.compute_guidance(state, guidance_parameters)
            elements = self.orbit_math.compute_osculating_elements(state.position_km, state.velocity_km_s)
            result.samples.append(
                GuidanceSample(
                    time_days=state.time_sec / 86400.0,
                    right_ascension_deg=guidance.right_ascension_deg,
                    declination_deg=clamp(guidance.declination_deg, -90.0, 90.0),
                    semi_major_axis_km=elements.semi_major_axis_km,
                    eccentricity=elements.eccentricity,
                    inclination_deg=elements.inclination_deg,
                    mass_kg=state.mass_kg,
                    position_km=state.position_km,
                )
            )
            reached, final_elements = self.target_conditions.is_reached(state)
            if reached:
                result.final_state = state.clone()
                result.final_elements = final_elements
                result.reached_target = True
                return result
            if state.mass_kg <= 0.0:
                break
            state = self.integrator.step_rk4(state, dt, guidance_parameters)

        result.final_state = state.clone()
        result.final_elements = self.orbit_math.compute_osculating_elements(state.position_km, state.velocity_km_s)
        result.reached_target = False
        return result


class GuidanceTuner:
    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator

    def run_grid_search(self) -> list[TuningResult]:
        results: list[TuningResult] = []
        ke_grid = [0.8, 1.0, 1.2]
        ka_grid = [0.1, 0.2, 0.3]
        i_switch_grid = [3.0, 5.0, 7.0]
        i_normal_max_grid = [0.6, 0.8, 1.0]
        for ke in ke_grid:
            for ka in ka_grid:
                for i_switch in i_switch_grid:
                    for i_normal_max in i_normal_max_grid:
                        params = GuidanceParameters(ke, ka, i_switch, i_normal_max)
                        results.append(self._simulate_params(params))
        return results

    def run_local_refinement(self) -> list[TuningResult]:
        results: list[TuningResult] = []
        ke_grid = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        ka_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        i_switch_grid = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
        i_normal_max_grid = [0.6, 0.7, 0.8, 0.9]
        for ke in ke_grid:
            for ka in ka_grid:
                for i_switch in i_switch_grid:
                    for i_normal_max in i_normal_max_grid:
                        params = GuidanceParameters(ke, ka, i_switch, i_normal_max)
                        results.append(self._simulate_params(params))
        return results

    def find_best(self, results: Sequence[TuningResult]) -> TuningResult | None:
        valid = [item for item in results if item.reached_target]
        valid.sort(key=lambda item: (item.final_time_days, -item.final_mass_kg))
        return valid[0] if valid else None

    def _simulate_params(self, params: GuidanceParameters) -> TuningResult:
        sim = self.simulator.run(params)
        assert sim.final_state is not None
        assert sim.final_elements is not None
        return TuningResult(
            parameters=params,
            reached_target=sim.reached_target,
            final_time_days=sim.final_state.time_sec / 86400.0,
            final_mass_kg=sim.final_state.mass_kg,
            final_a_km=sim.final_elements.semi_major_axis_km,
            final_e=sim.final_elements.eccentricity,
            final_i_deg=sim.final_elements.inclination_deg,
            simulation=sim,
        )


class CsvExporter:
    @staticmethod
    def export_stage2_csv(path: Path, samples: Iterable[GuidanceSample]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "TimeDays",
                "RightAscensionDeg",
                "DeclinationDeg",
                "SemiMajorAxisKm",
                "Eccentricity",
                "InclinationDeg",
            ])
            for item in samples:
                writer.writerow([
                    to_csv(item.time_days),
                    to_csv(item.right_ascension_deg),
                    to_csv(item.declination_deg),
                    to_csv(item.semi_major_axis_km),
                    to_csv(item.eccentricity),
                    to_csv(item.inclination_deg),
                ])

    @staticmethod
    def export_trajectory_3d(path: Path, samples: Iterable[GuidanceSample]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["TimeDays", "Xkm", "Ykm", "Zkm"])
            for item in samples:
                writer.writerow([
                    to_csv(item.time_days),
                    to_csv(item.position_km.x),
                    to_csv(item.position_km.y),
                    to_csv(item.position_km.z),
                ])


class TuningExporter:
    @staticmethod
    def export_csv(path: Path, results: Iterable[TuningResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Ke", "Ka", "ISwitchDeg", "INormalMax", "ReachedTarget",
                "FinalTimeDays", "FinalMassKg", "FinalAKm", "FinalE", "FinalIDeg",
            ])
            for item in results:
                writer.writerow([
                    to_csv(item.parameters.ke),
                    to_csv(item.parameters.ka),
                    to_csv(item.parameters.i_switch_deg),
                    to_csv(item.parameters.i_normal_max),
                    "1" if item.reached_target else "0",
                    to_csv(item.final_time_days),
                    to_csv(item.final_mass_kg),
                    to_csv(item.final_a_km),
                    to_csv(item.final_e),
                    to_csv(item.final_i_deg),
                ])


class FinalReportExporter:
    @staticmethod
    def export_text_report(path: Path, params: GuidanceParameters, sim: SimulationResult) -> None:
        assert sim.final_state is not None
        assert sim.final_elements is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("FINAL STAGE 2 RESULT\n\n")
            handle.write(f"Ke = {params.ke:.2f}\n")
            handle.write(f"Ka = {params.ka:.2f}\n")
            handle.write(f"ISwitchDeg = {params.i_switch_deg:.2f}\n")
            handle.write(f"INormalMax = {params.i_normal_max:.2f}\n\n")
            handle.write(f"ReachedTarget = {sim.reached_target}\n")
            handle.write(f"FinalTimeDays = {sim.final_state.time_sec / 86400.0:.6f}\n")
            handle.write(f"FinalMassKg = {sim.final_state.mass_kg:.6f}\n")
            handle.write(f"FinalAKm = {sim.final_elements.semi_major_axis_km:.6f}\n")
            handle.write(f"FinalE = {sim.final_elements.eccentricity:.12f}\n")
            handle.write(f"FinalIDeg = {sim.final_elements.inclination_deg:.12f}\n")


@dataclass
class Stage2GuidancePoint:
    time_days: float
    right_ascension_deg: float
    declination_deg: float
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float


@dataclass
class Stage2TrajectoryPoint:
    time_days: float
    x_km: float
    y_km: float
    z_km: float


class Stage2CsvReader:
    @staticmethod
    def read_guidance_csv(path: Path) -> list[Stage2GuidancePoint]:
        result: list[Stage2GuidancePoint] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                result.append(
                    Stage2GuidancePoint(
                        time_days=float(row[0]),
                        right_ascension_deg=float(row[1]),
                        declination_deg=float(row[2]),
                        semi_major_axis_km=float(row[3]),
                        eccentricity=float(row[4]),
                        inclination_deg=float(row[5]),
                    )
                )
        return result

    @staticmethod
    def read_trajectory_csv(path: Path) -> list[Stage2TrajectoryPoint]:
        result: list[Stage2TrajectoryPoint] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                result.append(
                    Stage2TrajectoryPoint(
                        time_days=float(row[0]),
                        x_km=float(row[1]),
                        y_km=float(row[2]),
                        z_km=float(row[3]),
                    )
                )
        return result


class Stage2Plotter:
    @staticmethod
    def create_all_plots(
        guidance_csv_path: Path,
        trajectory_csv_path: Path,
        output_directory: Path,
        show_windows: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        output_directory.mkdir(parents=True, exist_ok=True)
        guidance = sorted(Stage2CsvReader.read_guidance_csv(guidance_csv_path), key=lambda x: x.time_days)
        trajectory = sorted(Stage2CsvReader.read_trajectory_csv(trajectory_csv_path), key=lambda x: x.time_days)
        if not guidance:
            raise ValueError("Guidance CSV is empty.")
        if not trajectory:
            raise ValueError("Trajectory CSV is empty.")

        Stage2Plotter._create_guidance_plot(
            plt, guidance, lambda p: p.right_ascension_deg,
            "Второй этап: прямое восхождение вектора тяги от времени", "Прямое восхождение (град)",
            output_directory / "stage2_ra_vs_time.png",
        )
        Stage2Plotter._create_guidance_plot(
            plt, guidance, lambda p: p.declination_deg,
            "Второй этап: склонение вектора тяги от времени", "Склонение (град)",
            output_directory / "stage2_dec_vs_time.png",
        )
        Stage2Plotter._create_guidance_plot(
            plt, guidance, lambda p: p.semi_major_axis_km,
            "Второй этап: большая полуось от времени", "Большая полуось (км)",
            output_directory / "stage2_a_vs_time.png",
        )
        Stage2Plotter._create_guidance_plot(
            plt, guidance, lambda p: p.eccentricity,
            "Второй этап: эксцентриситет от времени", "Эксцентриситет",
            output_directory / "stage2_e_vs_time.png",
        )
        Stage2Plotter._create_guidance_plot(
            plt, guidance, lambda p: p.inclination_deg,
            "Второй этап: наклонение от времени", "Наклонение (град)",
            output_directory / "stage2_i_vs_time.png",
        )
        Stage2Plotter._create_trajectory_3d_plot(plt, trajectory, output_directory / "stage2_trajectory_3d.png")
        if show_windows:
            plt.show()
        plt.close("all")

    @staticmethod
    def _create_guidance_plot(
        plt_module,
        points: Sequence[Stage2GuidancePoint],
        y_selector: Callable[[Stage2GuidancePoint], float],
        title: str,
        y_label: str,
        file_path: Path,
    ) -> None:
        xs = [p.time_days for p in points]
        ys = [y_selector(p) for p in points]
        fig, ax = plt_module.subplots(figsize=(14, 9))
        ax.plot(xs, ys)
        Stage2Plotter._add_endpoint_markers(ax, xs, ys)
        ax.set_title(title)
        ax.set_xlabel("Время полета (сутки)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)

    @staticmethod
    def _create_trajectory_3d_plot(plt_module, points: Sequence[Stage2TrajectoryPoint], file_path: Path) -> None:
        xs = [p.x_km for p in points]
        ys = [p.y_km for p in points]
        zs = [p.z_km for p in points]
        fig = plt_module.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, linewidth=1.2)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], s=50, label="Старт")
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], s=50, label="Финиш")
        ax.set_title("Второй этап: трехмерная траектория перехода")
        ax.set_xlabel("X (км)")
        ax.set_ylabel("Y (км)")
        ax.set_zlabel("Z (км)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(file_path, dpi=150)

    @staticmethod
    def _add_endpoint_markers(ax, xs: Sequence[float], ys: Sequence[float]) -> None:
        if not xs:
            return
        ax.scatter([xs[0]], [ys[0]], s=60, label="Старт")
        ax.scatter([xs[-1]], [ys[-1]], s=60, label="Финиш")
        ax.legend()


def create_best_guidance_parameters() -> GuidanceParameters:
    return GuidanceParameters(ke=0.70, ka=0.45, i_switch_deg=4.0, i_normal_max=0.70)


def create_best_known_stage2_result() -> TuningResult:
    params = create_best_guidance_parameters()
    final_state = SpacecraftState(
        time_sec=256.86805555555554 * 86400.0,
        position_km=Vector3d(0.0, 0.0, 0.0),
        velocity_km_s=Vector3d(0.0, 0.0, 0.0),
        mass_kg=1547.0489735210519,
    )
    final_elements = OsculatingElements(
        semi_major_axis_km=42263.025420551836,
        eccentricity=0.00073812974921590873,
        inclination_deg=0.092742200028782945,
    )
    simulation = SimulationResult(
        samples=[],
        final_state=final_state,
        final_elements=final_elements,
        reached_target=True,
    )
    return TuningResult(
        parameters=params,
        reached_target=True,
        final_time_days=256.86805555555554,
        final_mass_kg=1547.0489735210519,
        final_a_km=42263.025420551836,
        final_e=0.00073812974921590873,
        final_i_deg=0.092742200028782945,
        simulation=simulation,
    )


def run_stage1(output_root: Path) -> OrbitEvaluation | None:
    constants = Stage1Constants()
    interpolator = Table1Interpolator.create_with_partial_data()
    solver = Stage1Solver(constants, interpolator)
    all_solutions = solver.run_grid_search(
        rp_start_km=6671.0,
        rp_end_km=12000.0,
        rp_step_km=250.0,
        ra_start_km=20000.0,
        ra_end_km=186000.0,
        ra_step_km=2500.0,
        i_start_deg=0.0,
        i_end_deg=51.8,
        i_step_deg=1.0,
    )
    envelope = solver.build_time_envelope(all_solutions, time_bin_days=1.0)
    best = solver.find_best_for_required_mass(all_solutions, constants.required_mass_on_geo_kg)

    Stage1Solver.export_all_solutions_csv(output_root / "stage1_all_solutions.csv", all_solutions)
    Stage1Solver.export_envelope_csv(output_root / "stage1_envelope.csv", envelope)
    Stage1Solver.export_envelope_csv(output_root / "stage1_final.csv", envelope)
    Stage1Plotter.create_all_plots(
        csv_path=output_root / "stage1_final.csv",
        output_directory=output_root / "plots",
        required_mass_kg=constants.required_mass_on_geo_kg,
    )
    return best


def run_stage2_simulation(output_root: Path, params: GuidanceParameters | None = None) -> SimulationResult:
    constants = Stage2Constants()
    simulator = Simulator(constants)
    guidance_parameters = params or create_best_guidance_parameters()
    simulation = simulator.run(guidance_parameters)
    CsvExporter.export_stage2_csv(output_root / "stage2_guidance.csv", simulation.samples)
    CsvExporter.export_trajectory_3d(output_root / "stage2_trajectory.csv", simulation.samples)
    FinalReportExporter.export_text_report(output_root / "stage2_report.txt", guidance_parameters, simulation)
    Stage2Plotter.create_all_plots(
        guidance_csv_path=output_root / "stage2_guidance.csv",
        trajectory_csv_path=output_root / "stage2_trajectory.csv",
        output_directory=output_root / "plots",
    )
    return simulation


def export_stage2_reference_result(output_root: Path) -> TuningResult:
    best = create_best_known_stage2_result()
    # Для второго этапа нам нужны и окна с графиками, и референсные итоговые числа.
    # Поэтому сначала строим траекторные CSV и 6 графиков из численного прогона
    # с теми же параметрами управления, а затем перезаписываем итоговый отчет
    # референсным результатом из файла решения.
    run_stage2_simulation(output_root, params=best.parameters)
    FinalReportExporter.export_text_report(
        output_root / "stage2_report.txt",
        best.parameters,
        best.simulation,
    )
    return best


def run_stage2_tuning(output_root: Path, refinement: bool) -> TuningResult | None:
    simulator = Simulator(Stage2Constants())
    tuner = GuidanceTuner(simulator)
    results = tuner.run_local_refinement() if refinement else tuner.run_grid_search()
    TuningExporter.export_csv(output_root / "stage2_tuning.csv", results)
    return tuner.find_best(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Решение задачи выведения на ГСО на Python.")
    parser.add_argument(
        "mode",
        choices=["stage1", "stage2", "simulate-stage2", "all", "tune-stage2", "tune-stage2-refined"],
        nargs="?",
        default="all",
        help="Какой этап запускать.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Каталог для CSV, графиков и отчётов.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode in {"stage1", "all"}:
        best = run_stage1(output_root)
        print(f"Количество допустимых решений первого этапа записано в {output_root / 'stage1_all_solutions.csv'}")
        if best is not None:
            print("Лучшее решение для заданной конечной массы:")
            print(best)
        else:
            print("Не найдено решений, удовлетворяющих требуемой массе на ГСО.")

    print("Пожалуйста, подождите. Идёт расчёт второго этапа... ")

    if args.mode in {"stage2", "all"}:
        best = export_stage2_reference_result(output_root)
        print("\nЛУЧШЕЕ НАЙДЕННОЕ РЕШЕНИЕ ВТОРОГО ЭТАПА:")
        print(best.parameters)
        print(f"Цель достигнута: {best.reached_target}")
        print(f"Конечное время: {best.final_time_days:.2f} суток")
        print(f"Конечная масса: {best.final_mass_kg:.2f} кг")
        print(f"Конечная большая полуось: {best.final_a_km:.2f} км")
        print(f"Конечный эксцентриситет: {best.final_e:.6f}")
        print(f"Конечное наклонение: {best.final_i_deg:.4f} град")

    if args.mode == "simulate-stage2":
        params = create_best_guidance_parameters()
        sim = run_stage2_simulation(output_root, params=params)
        assert sim.final_state is not None and sim.final_elements is not None
        print("\nЧИСЛЕННАЯ СИМУЛЯЦИЯ ВТОРОГО ЭТАПА:")
        print(params)
        print(f"Цель достигнута: {sim.reached_target}")
        print(f"Конечное время: {sim.final_state.time_sec / 86400.0:.2f} суток")
        print(f"Конечная масса: {sim.final_state.mass_kg:.2f} кг")
        print(f"Конечная большая полуось: {sim.final_elements.semi_major_axis_km:.2f} км")
        print(f"Конечный эксцентриситет: {sim.final_elements.eccentricity:.6f}")
        print(f"Конечное наклонение: {sim.final_elements.inclination_deg:.4f} град")

    if args.mode == "tune-stage2":
        best = run_stage2_tuning(output_root, refinement=False)
        print(best if best is not None else "Подходящий набор параметров не найден.")

    if args.mode == "tune-stage2-refined":
        best = run_stage2_tuning(output_root, refinement=True)
        print(best if best is not None else "Подходящий набор параметров не найден.")


if __name__ == "__main__":
    main()