# Value of Commercial Product Sales Data in Healthcare Prediction
## NHSX Analytics Unit - PhD Internship Project

### About the Project

This repository holds code for the NHSX Analytics Unit PhD internship project investigating the use of model class reliance to identify the value of including commerical sales data in respiratory death predictions by Elizabeth Dolan. <!-- state the work fits into your wider PhD thesis -->

[Project Description - Value of Commercial Product Sales Data in Healthcare Prediction](https://nhsx.github.io/nhsx-internship-projects/commercial-data-healthcare-predictions/)

_**Note:** No data, public or private are shared in this repository._

### Project Stucture

- The main code is found in the root of the repository (see Usage below for more information)
- The accompanying [report](./reports/report.pdf) is also available in the `reports` folder.  Results and Discussion to be published after peer review for journal submission.
- The Python libraries needed are listed in the requirements document. Please take note, you will need to go to https://github.com/gavin-s-smith/mcrforest to install the packages for MCR (Model Class Reliance).  You may need to install numpy and Cython before the mcrforest will install.  You will also need to install sci-kit learn version 0.24.2 in order to run the code "from sklearn.model_selection import TimeSeriesSplit" .  This TimeSeriesSplit version has the correct parameters to ensure no data leakage in the time series cross validation.   

### Built With

[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Shap] (https://shap.readthedocs.io/en/latest/index.html)
- [mcrforest] (https://github.com/gavin-s-smith/mcrforest)
### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsx/commercial-data-healthcare-predictions.git`

To create a suitable environment:
- ```python -m venv env``` or ```virtualenv -p /path/to/required/python/version .venv```
- `source .venv/bin/activate`
- (may need to) `pip install numpy` & `pip install Cython`
- `pip install git+https://github.com/gavin-s-smith/mcrforest`
- `pip install -r requirements.txt`

You may need to install pyscopg2 (https://www.psycopg.org/docs/install.html) which in turn can require gcc and additions to your PATH (https://stackoverflow.com/questions/5420789/how-to-install-psycopg2-with-pip-on-python).  

**Caveats for Apple Macbook Pro M1 users:**
Sklearn will not install using usual methods, the installation errors citing a build dependencies issue.
Use this line to install; `pip3 install -U --no-use-pep517 scikit-learn==0.24.2`
More information on this see https://github.com/scikit-learn/scikit-learn/issues/19137

### Usage

*Note: In it's current form this repoistory has been shared with fake data to allow the codes to run.  This data is randomly sampled from the same metadata features as the data but bears no resemblance to the ground truth data.*

run `Create_op_rf_for_mcr.py` to create a set of models to predict registered deaths from respiratory disease.  These models used commercial sales data and a wide range of other variables, which have shown associations with deaths from respiratory disease.

run `MCR_for_op_rf.py` to create explanations for the models by identifying the different impact variables inputted have on the models’ predictions, including commercial sales data.  This code implements the novel variable importance tool MCR for random forest regressor.

#### Dataset

Experiments are run against the: 
- ONS (Office for National Statistics) deaths registered weekly in England from diseases of the respiratory system (ICD-10 Coding J00–J99) https://digital.nhs.uk/services/primary-care-mortality-database;
- Weekly sales data from a UK high street retailer with stores distributed across the UK obtained through University partnership with UK Retailer;
- English indices of deprivation 2019 https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019;
- 2019 mid year population estimates based on the ONS census available at LTLA level and above https://www.nomisweb.co.uk/datasets/pestsyoala;
- Housing age data from the Valuation Office Agency https://data.cdrc.ac.uk/dataset/dwelling-ages-and-prices/resource/dwelling-age-group-counts-lsoa;
- Land use in England from 2018 live tables from the Department for Levelling Up, Housing and Communities and Ministry of Housing, Communities & Local Government https://www.gov.uk/government/statistical-data-sets/live-tables-on-land-use;
- Property type/ethnicity/household composition ONS 2011 census datasets https://www.nomisweb.co.uk/sources/census_2011;
- Weather. ERA5 data from European Centre for Medium-Range Weather Forecast https://copernicus.eu/.  
### Roadmap

See the [open issues](https://github.com/nhsx/commercial-data-healthcare-predictions/issues) for a list of proposed features (and known issues).  <!-- Add any known issues --> 

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk).

<!-- feel free to add other contacts here -->

### Acknowledgements

<!-- please acknowldege the data team here and the wider project -->
