import pandas as pd
import great_expectations as gx

# Load the GE context
context = gx.get_context()

# Load your pickle dataset
df = pd.read_pickle("/home/anish/airflow/dags/data/framingham_raw.pkl")

# Create a validator from pandas dataframe
validator = context.sources.pandas_default.read_dataframe(df)

# Add a simple expectation
validator.expect_column_values_to_not_be_null("age")

# Save expectation suite (new API: no overwrite_existing)
validator.save_expectation_suite(discard_failed_expectations=False)

print("✅ Expectation suite 'framingham_suite' created successfully!")
