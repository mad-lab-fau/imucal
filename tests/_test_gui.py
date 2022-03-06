from pathlib import Path

import pandas as pd

from imucal import ferraris_regions_from_interactive_plot

data = pd.read_csv(Path(__file__).parent.parent / "example_data/example_ferraris_session.csv").sort_values("n_samples")

section_data, section_list = ferraris_regions_from_interactive_plot(data)

print(section_data, section_list)
