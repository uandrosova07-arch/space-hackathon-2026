import math
import numpy as np

R_E     = 6371.0
R_GSO   = 42164.17
R_ENTRY = R_E + 100.0

# Стандартная атмосфера: экспоненциальная модель
H_SCALE = 8.5        # км, масштабная высота
RHO0    = 1.225e-9   # кг/м³ -> кг/км³ (переводим: 1.225 кг/м³ = 1.225e9 кг/км³)
# Удобнее держать всё в км и секундах:
RHO0_KM = 1.225e9    # кг/км³

def atm_density(h_km: float) -> float:
    """Плотность атмосферы на высоте h_km над поверхностью, кг/км³."""
    return RHO0_KM * math.exp(-h_km / H_SCALE)


def fall_kinematics(r0: float) -> tuple[float, float]:
    omega = 7.292e-5
    gm    = 398600.4415

    v_theta = omega * r0
    energy  = 0.5 * v_theta**2 - gm / r0
    a       = -gm / (2 * energy)
    L       = omega * r0**2
    e       = math.sqrt(max(0.0, 1 + 2 * energy * L**2 / gm**2))

    r_peri = a * (1 - e)
    if r_peri >= R_ENTRY:
        return math.inf, 0.0

    v_entry = math.sqrt(max(0.0, 2 * (energy + gm / R_ENTRY)))

    if r_peri >= R_E:
        T_orb   = 2 * math.pi * math.sqrt(a**3 / gm)
        cos_nu0 = (L**2 / (gm * r0) - 1) / e
        cos_nu0 = max(-1.0, min(1.0, cos_nu0))
        sin_nu0 = -math.sqrt(max(0.0, 1 - cos_nu0**2))
        cos_E0  = (e + cos_nu0) / (1 + e * cos_nu0)
        sin_E0  = math.sqrt(max(0.0, 1 - e**2)) * sin_nu0 / (1 + e * cos_nu0)
        E0      = math.atan2(sin_E0, cos_E0)
        M0      = E0 - e * sin_E0
        t_fall = T_orb / (2 * math.pi) * abs(M0)
    else:
        # Перигей внутри Земли — сегмент падает напрямую, не делая витка
        # t_fall считать численно или как четверть периода подземной орбиты
        # Грубая оценка: время свободного падения с высоты r0
        T_orb = 2 * math.pi * math.sqrt(a ** 3 / gm)
        t_fall = T_orb / 4  # грубо: четверть периода до перигея
        # Это завышено, но порядок правильный для оценки

    return t_fall, v_entry


def taper_ratio(r0_km: float, sigma_max_pa: float, rho_si: float) -> float:
    """Отношение A(r0) / A(R_GSO)."""
    gm    = 398600.4415e9   # м³/с²
    omega = 7.292e-5
    r     = r0_km * 1e3
    R_gso = R_GSO * 1e3
    phi   = gm * (1/r - 1/R_gso) - omega**2 / 2 * (R_gso**2 - r**2)
    return math.exp(rho_si * phi / sigma_max_pa)


def survived_fraction(v_entry_km_s: float,
                      d_mm: float,
                      rho_si: float,
                      H_c: float,
                      C_H: float = 0.1,
                      C_D: float = 1.0) -> float:
    """
    Оценка доли массы сегмента, пережившей абляцию.
    v_entry_km_s — скорость входа, км/с
    d_mm         — диаметр сечения, мм
    rho_si       — плотность материала, кг/м³
    H_c          — теплота сгорания/сублимации, Дж/кг
    """
    v = v_entry_km_s * 1e3          # м/с
    beta = rho_si * (d_mm * 1e-3) / (4 * C_D)   # кг/м²
    # Тепловой импульс на единицу площади миделя при входе:
    # Q = C_H * integral(rho_atm * v^3 / 2 dt) ~ C_H * beta * v^2 / 2
    # (грубая оценка: большая часть торможения происходит за ~1 масштабную высоту)
    Q_specific = C_H * v**2 / 2     # Дж/кг (на единицу массы)
    frac = math.exp(-Q_specific / H_c)
    return max(0.0, min(1.0, frac))


def tether_fem(n: int,
               d_gso_mm: float,
               sigma_max_gpa: float,
               rho_gcc: float,
               H_c_mj_kg: float) -> np.ndarray:
    """
    n             — число узлов МКЭ
    d_gso_mm      — диаметр троса на ГСО, мм
    sigma_max_gpa — предел прочности материала, ГПа
    rho_gcc       — плотность материала, г/см³
    H_c_mj_kg     — теплота сгорания/сублимации, МДж/кг

    Возвращает массив (n-1, 7):
      [r_mid, d_mid, lambda, m_seg, t_fall, v_entry, m_impact]
    где r_mid — середина сегмента, m_impact — масса, долетевшая до земли.
    """
    sigma_pa = sigma_max_gpa * 1e9
    rho_si   = rho_gcc * 1e3
    H_c      = H_c_mj_kg * 1e6
    A_gso    = math.pi * (d_gso_mm * 1e-3)**2 / 4

    nodes = np.linspace(R_E, R_GSO, n)
    dl    = nodes[1] - nodes[0]   # длина одного сегмента, км

    results = []
    for i in range(n - 1):
        r_mid = 0.5 * (nodes[i] + nodes[i + 1])

        # Геометрия сечения
        ratio = taper_ratio(r_mid, sigma_pa, rho_si)
        A     = A_gso * ratio                        # м²
        d_mm  = math.sqrt(4 * A / math.pi) * 1e3    # мм

        # Линейная плотность и масса сегмента
        lam   = rho_si * A                           # кг/м
        m_seg = lam * dl * 1e3                       # кг (dl в км -> м)

        # Кинематика
        t_fall, v_entry = fall_kinematics(r_mid)

        # Абляция
        if v_entry > 0 and t_fall < math.inf:
            frac     = survived_fraction(v_entry, d_mm, rho_si, H_c)
            m_impact = m_seg * frac
        else:
            frac     = 0.0
            m_impact = 0.0

        results.append((r_mid, d_mm, lam, m_seg,
                        t_fall, v_entry, m_impact))

    return np.array(results)


def impact_energy(m_impact_kg: float, v_entry_km_s: float) -> float:
    """Кинетическая энергия удара, Дж."""
    v = v_entry_km_s * 1e3
    return 0.5 * m_impact_kg * v**2

def qualify_impact(total_energy_gdj: float,
                   total_impact_kg: float,
                   n_segments: int,
                   tether_length_km: float) -> str:
    """
    Квалификация последствий падения троса.
    total_energy_gdj  — суммарная кинетическая энергия удара, ГДж
    total_impact_kg   — суммарная масса, долетевшая до земли, кг
    n_segments        — число сегментов, долетевших до земли
    tether_length_km  — длина упавшей части троса, км
    """

    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("КВАЛИФИКАЦИЯ ПОСЛЕДСТВИЙ")
    lines.append("=" * 60)

    # --- Энергетическая шкала ---
    HIROSHIMA   = 63        # ГДж
    TUNGUSKA    = 10_000    # ГДж
    EARTHQUAKE8 = 630_000   # ГДж

    lines.append("\nЭнергетическая шкала:")
    lines.append(f"  Суммарная энергия удара : {total_energy_gdj:>12.1f} ГДж")
    lines.append(f"  Бомба «Малыш» (Хиросима): {HIROSHIMA:>12.1f} ГДж")
    lines.append(f"  Тунгусский метеорит     : {TUNGUSKA:>12.1f} ГДж")
    lines.append(f"  Землетрясение М8.0      : {EARTHQUAKE8:>12.1f} ГДж")
    lines.append(f"  Эквивалент в Хиросимах  : {total_energy_gdj/HIROSHIMA:>12.1f}")
    lines.append(f"  Эквивалент в Тунгусках  : {total_energy_gdj/TUNGUSKA:>12.4f}")

    # --- Распределение энергии по площади ---
    # Трос падает вдоль экватора, ширина зоны поражения ~10 км
    STRIP_WIDTH_KM = 10.0
    area_km2 = tether_length_km * STRIP_WIDTH_KM
    area_m2  = area_km2 * 1e6
    energy_density = total_energy_gdj * 1e9 / area_m2   # Дж/м²

    lines.append(f"\nРаспределение по площади:")
    lines.append(f"  Длина зоны падения      : {tether_length_km:>12.0f} км")
    lines.append(f"  Площадь зоны (~10 км шир): {area_km2:>11.0f} км²")
    lines.append(f"  Плотность энергии       : {energy_density:>12.1f} Дж/м²")

    # Для сравнения: ядерный взрыв ~1 Мт на площади ~100 км² = ~4e10 Дж/м²
    # Сильный пожар ~1e6 Дж/м², взрывная волна разрушения ~1e4 Дж/м²
    DESTRUCTION_THRESHOLD = 1e4   # Дж/м² — порог разрушения зданий
    FIRE_THRESHOLD        = 1e6   # Дж/м² — порог воспламенения

    # --- Масса ---
    lines.append(f"\nМасса:")
    lines.append(f"  Долетело до земли       : {total_impact_kg:>12.1f} кг")
    lines.append(f"  Число упавших сегментов : {n_segments:>12d}")
    if n_segments > 0:
        lines.append(f"  Средняя масса сегмента  : "
                     f"{total_impact_kg/n_segments:>12.1f} кг")

    # --- Вердикт ---
    lines.append("\n" + "-" * 60)
    lines.append("ВЕРДИКТ:")

    if total_energy_gdj < 10:
        level = "ЛОКАЛЬНЫЙ ИНЦИДЕНТ"
        description = ("Энергия сопоставима с крупной промышленной аварией. "
                       "Жертвы возможны в зоне падения, глобальных последствий нет.")
    elif total_energy_gdj < 100:
        level = "КРУПНАЯ КАТАСТРОФА"
        description = ("Энергия порядка нескольких ядерных бомб. "
                       "Значительные разрушения вдоль трассы падения (~экватор). "
                       "Региональные последствия.")
    elif total_energy_gdj < TUNGUSKA:
        level = "ОЧЕНЬ КРУПНАЯ КАТАСТРОФА"
        description = ("Энергия от 100 до 10 000 ГДж. Сопоставимо с Тунгусским "
                       "событием. Разрушения на протяжённой полосе вдоль экватора. "
                       "Возможен региональный климатический эффект.")
    elif total_energy_gdj < 10 * TUNGUSKA:
        level = "ГЛОБАЛЬНАЯ КАТАСТРОФА"
        description = ("Энергия превышает Тунгусское событие. Разрушения на тысячах "
                       "километров экватора. Глобальный климатический эффект вероятен.")
    else:
        level = "КАТАСТРОФА ЦИВИЛИЗАЦИОННОГО МАСШТАБА"
        description = ("Энергия на уровне крупнейших природных катастроф в истории. "
                       "Глобальные последствия неизбежны.")

    lines.append(f"  Уровень: {level}")
    lines.append(f"  {description}")

    if energy_density > FIRE_THRESHOLD:
        lines.append("  ! Плотность энергии достаточна для воспламенения в зоне падения.")
    elif energy_density > DESTRUCTION_THRESHOLD:
        lines.append("  ! Плотность энергии достаточна для разрушения зданий.")
    else:
        lines.append("  Плотность энергии ниже порога массовых разрушений.")

    lines.append("=" * 60)
    return "\n".join(lines)

if __name__ == "__main__":
    N            = int(input("Число узлов: "))
    d_gso        = float(input("Диаметр на ГСО (мм): "))
    sigma        = float(input("Предел прочности (ГПа): "))
    rho          = float(input("Плотность (г/см³): "))
    H_c          = float(input("Теплота сгорания (МДж/кг): "))

    data = tether_fem(N, d_gso, sigma, rho, H_c)

    print(f"\n{'r (км)':>10} {'d (мм)':>10} {'λ (кг/м)':>10} "
          f"{'m_сег (кг)':>12} {'t_fall (ч)':>11} "
          f"{'v (км/с)':>9} {'m_удар (кг)':>12} {'E (МДж)':>10}")
    print("-" * 96)

    total_mass   = 0.0
    total_impact = 0.0
    total_energy = 0.0

    for r, d, lam, m_seg, t, v, m_imp in data:
        total_mass   += m_seg
        total_impact += m_imp
        E = impact_energy(m_imp, v) / 1e6   # МДж
        total_energy += E
        t_h = t / 3600 if t < math.inf else math.inf
        print(f"{r:10.0f} {d:10.3f} {lam:10.4f} "
              f"{m_seg:12.1f} {t_h:11.2f} "
              f"{v:9.3f} {m_imp:12.1f} {E:10.1f}")

    print("-" * 96)
    print(f"{'Итого':>10}  {'':>10}  {'':>10}  "
          f"{total_mass:12.1f}  {'':>11}  {'':>9}  "
          f"{total_impact:12.1f}  {total_energy:10.1f}")
    print(f"\nДоля массы, долетевшей до земли: "
          f"{100*total_impact/total_mass:.2f}%")
    print(f"Суммарная энергия удара: {total_energy/1e3:.2f} ГДж")

    # Считаем параметры для квалификации
    fallen_mask = data[:, 6] > 0  # сегменты, долетевшие до земли
    n_fallen = int(fallen_mask.sum())
    fallen_r = data[fallen_mask, 0]
    tether_length = (fallen_r.max() - fallen_r.min()
                     if n_fallen > 1 else 0.0)

    print(qualify_impact(
        total_energy_gdj=total_energy / 1e3,
        total_impact_kg=total_impact,
        n_segments=n_fallen,
        tether_length_km=tether_length
    ))