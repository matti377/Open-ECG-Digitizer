/opt/homebrew/bin/python3.12 -m venv openecg-env
source openecg-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Image in correct folder

digitize: python3 -m src.digitize --config inference_wrapper.yml

digitalisation image weisen

python3 scripts/plot_12lead.py output_data/digitalisation/ecg_timeseries_canonical.csv

python3 scripts/csv_to_wfdb.py output_data/digitalisation/ecg_timeseries_canonical.csv

.dat an .hea weisen

python3 scripts/dat_plotter.py output_data/wfdb/record

python3 scripts/analyse.py output_data/wfdb/record