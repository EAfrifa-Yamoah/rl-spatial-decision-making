"""
Reference classification for the v4 corpus.

Classifies the 96 references in the v4 manuscript into applied-domain and
foundational categories. Each reference is assigned to exactly one domain
based on its primary citation context.

v4 adds 18 references over v3 (78 -> 96), addressing reviewer 1's request
for expanded GIScience, remote sensing, transportation MARL, and spatial
resource allocation coverage.

Run:
    python -m docs.reference_classification
"""
from collections import Counter

CLASSIFICATION = {
    # ---- Robot navigation / manipulation ----
    "Hwangbo et al. 2019": "robot_navigation",
    "Wang et al. 2023": "robot_navigation",
    "Richter et al. 2019": "robot_navigation",

    # ---- Autonomous vehicles ----
    "Kendall et al. 2019": "autonomous_vehicles",
    "Kiran et al. 2022": "autonomous_vehicles",
    "Sallab et al. 2017": "autonomous_vehicles",
    "Shalev-Shwartz et al. 2016": "autonomous_vehicles",

    # ---- UAV / swarm coordination ----
    "Xu C. et al. 2022": "uav_swarm",

    # ---- Sensor placement / adaptive sensing ----
    "Krause et al. 2008": "sensor_placement",
    "MacKay 1992": "sensor_placement",
    "Singh et al. 2009": "sensor_placement",
    "Hsieh et al. 2015": "sensor_placement",
    "Flaspohler et al. 2019": "sensor_placement",

    # ---- Environmental monitoring ----
    "Lin Y et al. 2023": "env_monitoring",
    "Liao et al. 2017": "env_monitoring",

    # ---- Conservation & wildlife protection ----
    "Lapeyrolerie et al. 2022": "conservation_wildlife",
    "Marrec & Boettiger 2023": "conservation_wildlife",
    "Ferrer-Mestres et al. 2021": "conservation_wildlife",
    "Yang et al. 2014": "conservation_wildlife",
    "Fang Stone Tambe 2015": "conservation_wildlife",
    "Xu L. et al. 2021": "conservation_wildlife",

    # ---- Forestry / wildfire management ----
    "Subramanian & Crowley 2018": "forestry_wildfire",
    "Altamimi et al. 2022": "forestry_wildfire",
    "Murray et al. 2024": "forestry_wildfire",

    # ---- Water / reservoir operations ----
    "Wu et al. 2024": "water_reservoir",
    "Chaipipattanawong et al. 2025": "water_reservoir",
    "Ficchi et al. 2016": "water_reservoir",  # NEW v4

    # ---- GIS-RL integration ----
    "Gu et al. 2020": "gis_rl",
    "Saber et al. 2025": "gis_rl",
    "Theodoropoulos et al. 2023": "gis_rl",
    "Zhang D et al. 2024": "gis_rl",
    "Duan et al. 2020": "gis_rl",  # NEW v4

    # ---- Remote sensing (new category in v4) ----
    "Liu Y Zhong Shi Zhang 2024": "remote_sensing",  # NEW v4
    "Ayush et al. 2020": "remote_sensing",  # NEW v4
    "Lacoste et al. 2023": "remote_sensing",  # NEW v4

    # ---- Transportation / MARL (strengthened in v4) ----
    "Liu Chen Chen 2023": "transportation",  # NEW v4
    "Wang STMARL 2022": "transportation",  # NEW v4
    "Chu et al. 2020": "transportation",  # NEW v4
    "Hu & Li 2024": "transportation",  # NEW v4

    # ---- Urban / land use / facility planning ----
    "Longley et al. 2015": "urban_planning",
    "Church & Murray 2009": "urban_planning",

    # ---- Foundational RL (algorithms, theory, evaluation, surveys) ----
    "Alshiekh et al. 2018": "foundational",
    "Altman 1999": "foundational",
    "Amodei et al. 2016": "foundational",
    "Arulkumaran et al. 2017": "foundational",
    "Battaglia et al. 2018": "foundational",
    "Chebotar et al. 2019": "foundational",
    "Duan Y et al. 2016": "foundational",
    "Finn Abbeel Levine 2017": "foundational",
    "Foerster et al. 2018": "foundational",
    "Fujimoto van Hoof Meger 2018": "foundational",
    "Haarnoja et al. 2018": "foundational",
    "Hafner et al. 2020": "foundational",
    "Hausknecht & Stone 2015": "foundational",
    "Hessel et al. 2018": "foundational",
    "Janner et al. 2019": "foundational",
    "Jumper et al. 2021": "foundational",
    "Kumar et al. 2020": "foundational",
    "Leike et al. 2017": "foundational",
    "Levine et al. 2020": "foundational",
    "Lillicrap et al. 2016": "foundational",
    "Littman 1994": "foundational",
    "Lowe et al. 2017": "foundational",
    "Mnih et al. 2015": "foundational",
    "Nachum et al. 2018": "foundational",
    "Ng Harada Russell 1999": "foundational",
    "Parisotto et al. 2020": "foundational",
    "Peng et al. 2018": "foundational",
    "Rashid et al. 2020": "foundational",
    "Ray Achiam Amodei 2019": "foundational",
    "Rummery & Niranjan 1994": "foundational",
    "Sanchez-Gonzalez et al. 2020": "foundational",
    "Schrittwieser et al. 2020": "foundational",
    "Schulman et al. 2017": "foundational",
    "Silver et al. 2016": "foundational",
    "Sutton 1991": "foundational",
    "Sutton & Barto 2018": "foundational",
    "Sutton Precup Singh 1999": "foundational",
    "Tampuu et al. 2017": "foundational",
    "Taylor & Stone 2009": "foundational",
    "Tobin et al. 2017": "foundational",
    "Vaswani et al. 2017": "foundational",
    "van Hasselt Guez Silver 2016": "foundational",
    "Vezhnevets et al. 2017": "foundational",
    "Watkins & Dayan 1992": "foundational",
    "Williams 1992": "foundational",
    "Zhu et al. 2023": "foundational",

    # ---- Foundational added in v4 ----
    "Ada et al. 2024": "foundational",           # diffusion policies OOD
    "Agarwal et al. 2021": "foundational",       # statistical precipice
    "Cobbe et al. 2019": "foundational",         # quantifying generalisation
    "Kirk et al. 2023": "foundational",          # zero-shot generalisation survey
    "Nguyen et al. 2020": "foundational",        # deep MARL review
    "Zhang K Yang Basar 2021": "foundational",   # MARL selective overview
    "Meyer et al. 2019": "foundational",         # spatial predictor selection
    "Ploton et al. 2020": "foundational",        # spatial validation
    "Anselin 1995": "foundational",              # LISA
    "Openshaw 1984": "foundational",             # MAUP
    "Tobler 1970": "foundational",               # first law of geography
}


def _report() -> None:
    counts = Counter(CLASSIFICATION.values())
    total = sum(counts.values())
    applied = total - counts["foundational"]
    foundational = counts["foundational"]

    print(f"Total references classified: {total}")
    print()
    print("By domain (descending count):")
    for domain, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:22s} {count}")
    print()
    print(f"Applied domain refs:      {applied:>3d}  ({applied / total * 100:.0f}%)")
    print(f"Foundational refs:        {foundational:>3d}  ({foundational / total * 100:.0f}%)")


if __name__ == "__main__":
    _report()
