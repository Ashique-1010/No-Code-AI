
# This notebook contains a Scikit-learn representation of AutoAI pipeline. This notebook introduces commands for retrieving data, training the model, and testing the model. 
# 
# Some familiarity with Python is helpful. This notebook uses Python 3.10 and scikit-learn 1.1.1.

# ## Notebook goals
# 
# -  Scikit-learn pipeline definition
# -  Pipeline training 
# -  Pipeline evaluation
# 

# ## Package installation
# Before you use the sample code in this notebook, install the following packages:
#  - ibm-watsonx-ai,
#  - autoai-libs,
#  - scikit-learn,
#  - snapml


get_ipython().system('pip install ibm-watsonx-ai | tail -n 1')
get_ipython().system('pip install autoai-libs==1.16.2 | tail -n 1')
get_ipython().system('pip install scikit-learn==1.1.1 | tail -n 1')
get_ipython().system("pip install -U 'lale>=0.7,<0.8' | tail -n 1")
get_ipython().system('pip install snapml==1.13.2 | tail -n 1')


# Filter warnings for this notebook.


import warnings

warnings.filterwarnings('ignore')



from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation

training_data_references = [
    DataConnection(
        data_asset_id='888e3af4-3ad8-4ff4-8e0f-5c7847d3799b'
    ),
]
training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/0e27ae26-e80a-4fac-a7ec-d7546d39c940/wml_data/fc162f30-50b2-4a6b-a3a6-97d7d3fc3ebe/data/automl',
        model_location='auto_ml/0e27ae26-e80a-4fac-a7ec-d7546d39c940/wml_data/fc162f30-50b2-4a6b-a3a6-97d7d3fc3ebe/data/automl/model.zip',
        training_status='auto_ml/0e27ae26-e80a-4fac-a7ec-d7546d39c940/wml_data/fc162f30-50b2-4a6b-a3a6-97d7d3fc3ebe/training-status.json'
    )
)


# The following cell contains input parameters provided to run the AutoAI experiment in Watson Studio.



experiment_metadata = dict(
    prediction_type='binary',
    prediction_column='Risk',
    holdout_size=0.1,
    scoring='accuracy',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    include_only_estimators=['RandomForestClassifierEstimator', 'DecisionTreeClassifierEstimator', 'LogisticRegressionEstimator', 'ExtraTreesClassifierEstimator', 'XGBClassifierEstimator', 'LGBMClassifierEstimator', 'SnapDecisionTreeClassifierEstimator', 'SnapRandomForestClassifierEstimator', 'SnapBoostingMachineClassifierEstimator', 'SnapLogisticRegressionEstimator', 'SnapSVMClassifierEstimator', 'GradientBoostingClassifierEstimator'],
    deployment_url='https://us-south.ml.cloud.ibm.com',
    project_id='01cdde67-0ab8-4969-9406-d4221ddddeed',
    train_sample_columns_index_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19],
    positive_label='No Risk',
    drop_duplicates=True,
    include_batched_ensemble_estimators=[],
    feature_selector_mode='off'
)


# ## Watson Machine Learning connection
# 
# This cell defines the credentials required to work with the Watson Machine Learning service.
# 
# **Action**: Provide the IBM Cloud apikey, For details, see [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey).



api_key = 'PUT_YOUR_APIKEY_HERE'




wml_credentials = {
    "apikey": api_key,
    "url": experiment_metadata['deployment_url']
}



from ibm_watsonx_ai import APIClient

wml_client = APIClient(wml_credentials)

if 'space_id' in experiment_metadata:
    wml_client.set.default_space(experiment_metadata['space_id'])
else:
    wml_client.set.default_project(experiment_metadata['project_id'])
    
training_data_references[0].set_client(wml_client)


# # Pipeline inspection

# ## Read training data
# 
# Retrieve training dataset from AutoAI experiment as pandas DataFrame.
# 
# **Note**: If reading data results in an error, provide data as Pandas DataFrame object, for example, reading .CSV file with `pandas.read_csv()`. 
# 
# It may be necessary to use methods for initial data pre-processing like: e.g. `DataFrame.dropna()`, `DataFrame.drop_duplicates()`, `DataFrame.sample()`.
# 



train_X, test_X, train_y, test_y = training_data_references[0].read(experiment_metadata=experiment_metadata, with_holdout_split=True, use_flight=False)


# ## Create pipeline
# In the next cell, you can find the Scikit-learn definition of the selected AutoAI pipeline.

# #### Import statements.


from autoai_libs.transformers.exportable import ColumnSelector
from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import CompressStrings
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumpyReplaceUnknownValues
from autoai_libs.transformers.exportable import boolean2float
from autoai_libs.transformers.exportable import CatImputer
from autoai_libs.transformers.exportable import CatEncoder
import numpy as np
from autoai_libs.transformers.exportable import float32_transform
from sklearn.pipeline import make_pipeline
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from sklearn.pipeline import make_union
from autoai_libs.transformers.exportable import NumpyPermuteArray
from autoai_libs.cognito.transforms.transform_utils import TAM
from sklearn.decomposition import PCA
from autoai_libs.cognito.transforms.transform_utils import FS1
from autoai_libs.cognito.transforms.transform_utils import TA2
import autoai_libs.utils.fc_methods
from snapml import SnapBoostingMachineClassifier



# #### Pre-processing & Estimator.


column_selector_0 = ColumnSelector(
    columns_indices_list=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19,
    ]
)
numpy_column_selector_0 = NumpyColumnSelector(
    columns=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
)
compress_strings = CompressStrings(
    compress_type="hash",
    dtypes_list=[
        "char_str", "int_num", "char_str", "char_str", "char_str", "char_str",
        "int_num", "char_str", "char_str", "int_num", "char_str", "int_num",
        "char_str", "char_str", "int_num", "char_str", "int_num", "char_str",
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
    misslist_list=[
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
        [],
    ],
)
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(
    filling_values=float("nan"), missing_values=[]
)
numpy_replace_unknown_values = NumpyReplaceUnknownValues(
    filling_values=float("nan"),
    filling_values_list=[
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        float("nan"), float("nan"), float("nan"),
    ],
    missing_values_reference_list=["", "-", "?", float("nan")],
)
cat_imputer = CatImputer(
    missing_values=float("nan"),
    sklearn_version_family="1",
    strategy="most_frequent",
)
cat_encoder = CatEncoder(
    dtype=np.float64,
    handle_unknown="error",
    sklearn_version_family="1",
    encoding="ordinal",
    categories="auto",
)
pipeline_0 = make_pipeline(
    column_selector_0,
    numpy_column_selector_0,
    compress_strings,
    numpy_replace_missing_values_0,
    numpy_replace_unknown_values,
    boolean2float(),
    cat_imputer,
    cat_encoder,
    float32_transform(),
)
column_selector_1 = ColumnSelector(
    columns_indices_list=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19,
    ]
)
numpy_column_selector_1 = NumpyColumnSelector(columns=[4])
float_str2_float = FloatStr2Float(
    dtypes_list=["int_num"], missing_values_reference_list=[]
)
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(
    filling_values=float("nan"), missing_values=[]
)
num_imputer = NumImputer(missing_values=float("nan"), strategy="median")
opt_standard_scaler = OptStandardScaler(use_scaler_flag=False)
pipeline_1 = make_pipeline(
    column_selector_1,
    numpy_column_selector_1,
    float_str2_float,
    numpy_replace_missing_values_1,
    num_imputer,
    opt_standard_scaler,
    float32_transform(),
)
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(
    axis=0,
    permutation_indices=[
        0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4,
    ],
)
tam = TAM(
    tans_class=PCA(),
    name="pca",
    col_names=[
        "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose",
        "LoanAmount", "ExistingSavings", "EmploymentDuration",
        "InstallmentPercent", "Sex", "OthersOnLoan",
        "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans",
        "Housing", "ExistingCreditsCount", "Job", "Dependents",
        "ForeignWorker",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"),
    ],
)
fs1_0 = FS1(
    cols_ids_must_keep=range(0, 19),
    additional_col_count_to_keep=15,
    ptype="classification",
)
ta2 = TA2(
    fun=np.add,
    name="sum",
    datatypes1=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints1=[autoai_libs.utils.fc_methods.is_not_categorical],
    datatypes2=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints2=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "CheckingStatus", "LoanDuration", "CreditHistory", "LoanPurpose",
        "LoanAmount", "ExistingSavings", "EmploymentDuration",
        "InstallmentPercent", "Sex", "OthersOnLoan",
        "CurrentResidenceDuration", "OwnsProperty", "Age", "InstallmentPlans",
        "Housing", "ExistingCreditsCount", "Job", "Dependents",
        "ForeignWorker", "pca_0", "pca_1", "pca_2", "pca_4", "pca_5", "pca_6",
        "pca_8", "pca_9", "pca_10", "pca_12", "pca_13", "pca_14", "pca_15",
        "pca_16", "pca_18",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"),
    ],
)
fs1_1 = FS1(
    cols_ids_must_keep=range(0, 19),
    additional_col_count_to_keep=15,
    ptype="classification",
)
snap_boosting_machine_classifier = SnapBoostingMachineClassifier(
    class_weight="balanced", gpu_ids=[0], random_state=33
)


# #### Pipeline.


pipeline = make_pipeline(
    union,
    numpy_permute_array,
    tam,
    fs1_0,
    ta2,
    fs1_1,
    snap_boosting_machine_classifier,
)


# ## Train pipeline model
# ### Define scorer from the optimization metric
# This cell constructs the cell scorer based on the experiment metadata.


from sklearn.metrics import get_scorer

scorer = get_scorer(experiment_metadata['scoring'])


# ### Fit pipeline model
# In this cell, the pipeline is fitted.


pipeline.fit(train_X.values, train_y.values.ravel());


# ## Test pipeline model
# Score the fitted pipeline with the generated scorer using the holdout dataset.

score = scorer(pipeline, test_X.values, test_y.values)
print(score)



pipeline.predict(test_X.values[:5])

# ## Store the model 
# In this section you will learn how to store the trained model.



model_metadata = {
    wml_client.repository.ModelMetaNames.NAME: 'P4 - Pretrained AutoAI pipeline'
}

stored_model_details = wml_client.repository.store_model(model=pipeline, meta_props=model_metadata, experiment_metadata=experiment_metadata)


# Inspect the stored model details.

print(stored_model_details)


# ## Create online deployment

# You can use commands bellow to promote the model to space and create online deployment (web service).
# 
# ### Working with spaces
# 
# In this section you will specify a deployment space for organizing the assets for deploying and scoring the model. If you do not have an existing space, you can use [Deployment Spaces Dashboard](https://dataplatform.cloud.ibm.com/ml-runtime/spaces?context=cpdaas) to create a new space, following these steps:
# 
# - Click **New Deployment Space**.
# - Create an empty space.
# - Select Cloud Object Storage.
# - Select Watson Machine Learning instance and press **Create**.
# - Copy `space_id` and paste it below.
# 
# **Tip**: You can also use the API to prepare the space for your work. Learn more [here](https://github.com/IBM/watson-machine-learning-samples/blob/master/notebooks/python_sdk/instance-management/Space%20management.ipynb).
# 
# **Action**: Assign or update space ID below.
space_id = "PUT_YOUR_SPACE_ID_HERE"

model_id = wml_client.spaces.promote(asset_id=stored_model_details["metadata"]["id"], source_project_id=experiment_metadata["project_id"], target_space_id=space_id)
# #### Prepare online deployment
wml_client.set.default_space(space_id)

deploy_meta = {
        wml_client.deployments.ConfigurationMetaNames.NAME: "Incrementally trained AutoAI pipeline",
        wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
    }

deployment_details = wml_client.deployments.create(artifact_uid=model_id, meta_props=deploy_meta)
deployment_id = wml_client.deployments.get_id(deployment_details)
# #### Test online deployment
import pandas as pd

scoring_payload = {
    "input_data": [{
        'values': pd.DataFrame(test_X[:5])
    }]
}

wml_client.deployments.score(deployment_id, scoring_payload)
# ### Deleting deployment
# You can delete the existing deployment by calling the `wml_client.deployments.delete(deployment_id)` command.
# To list the existing web services, use `wml_client.deployments.list()`.


# # Summary and next steps
# You successfully completed this notebook!
# You learned how to use AutoAI pipeline definition to train the model.
# Check out our [Online Documentation](https://www.ibm.com/cloud/watson-studio/autoai) for more samples, tutorials, documentation, how-tos, and blog posts.
 
# **Note:** The auto-generated notebooks are subject to the International License Agreement for Non-Warranted Programs (or equivalent) and License Information document for Watson Studio Auto-generated Notebook (License Terms), such agreements located in the link below. Specifically, the Source Components and Sample Materials clause included in the License Information document for Watson Studio Auto-generated Notebook applies to the auto-generated notebooks.   
# By downloading, copying, accessing, or otherwise using the materials, you agree to the <a href="https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF">License Terms</a>
