Project workflow of Loan risk assessment on German credit Dataset using IBM Watsonx
Gradient Boosting Algorithm is used to train the data, and the pipeline with highest accuracy is selected.

1. Create Cloud object storage.

2. Create Watsonx instance:

   * Set location to dallas
   * Launch in ibm cloud pak for data

3. Provision watsonx for Machine Learning:

   * Set Dallas as region
   * Create

4. Redirected to watsonx studio:

   * Create new project:
     * Select name and storage
     * Upload assets(dataset)

   * Create new asset:(The ML model)
     * Automated builders:
       * Select AutoAI:
         * Associate a ML instance:
           * Select the Watsonx instance you have created
         * Create the AI
         * Add Data sources from project:
           * Select the data assets
           * Time series forecast is not needed
           * Select the required feature to predict(Risk)
             * Update Experiment settings:
               * Set prediction type: Binary classification
               * prediction:
                 * using accuracy as optimization metric
                 * select gradient boosting classifier
                 * data source:
                   * select train test split: 90:10
                   * deselect unwanted features
           * Run experiment
           * Training results are displayed with the metric
           * Select the best pipeline
             * Evaluate results, confusion matrix
             * Save as notebook
             * View in project



